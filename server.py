from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.eval_utils import (
    ModelConfig,
    all_model_cfg,
    generate,
    load_video,
    setup_eval_logging,
)
import tempfile
import requests
from datetime import datetime
import torchaudio
import torch
from pathlib import Path
import logging
import os
import random
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from supabase import create_client, Client
import av
import uuid
import asyncio
from typing import Optional
from enum import Enum

load_dotenv()

app = FastAPI(title="RhythmAI API", version="1.0.0")

origins = [
    "*",  # for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prompt = """
Provide a detailed description of the background music the video should feature based on user's request. The musical description should cover the following five elements:

1.  **Genre and Style:** Suggest a specific genre (e.g., Lo-Fi, Cinematic Orchestral, Acoustic Folk, Synthwave).
2.  **Instrumentation:** Specify the core instruments (e.g., "Main melody on piano with a cello backing," "Simple drum loop with ambient synth pads," "Acoustic guitar and light percussion").
3.  **Tempo and Dynamics:** Describe the speed (BPM) and loudness changes (e.g., "Medium tempo, 100 BPM, starting softly and building dynamics at the 0:45 mark.").
4.  **Mood and Key:** State the emotional effect and suggest a musical key (e.g., "The mood should be hopeful and slightly melancholic, suggesting a minor key like C minor.").
5.  **Placement Notes:** Provide guidance on where the music should be most prominent or subtle (e.g. "Use a climactic chord progression for the final 15 seconds.").

User's request: {user_request}
Output background music description in English.
"""

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Create Supabase client
supabase: Client = create_client(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)


class GenerateRequest(BaseModel):
    video_url: str
    user_request: str


class PingResponse(BaseModel):
    status: str
    message: str


class GenerateResponse(BaseModel):
    job_id: str


class JobStatusEnum(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class AudioResult(BaseModel):
    prompt: str
    audio_url: str
    index: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    completed_count: int
    total_count: int
    results: list[AudioResult] = []
    error: Optional[str] = None


# In-memory job storage
jobs_db = {}


def get_bgm_description(video_url: str, user_request: str) -> str:
    completion = client.chat.completions.create(
        model="ernie-4.5-vl-28b-a3b",
        temperature=0.8,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.format(user_request=user_request)},
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_url,
                            "fps": 1,
                        },
                    },
                ],
            }
        ],
    )
    return completion.choices[0].message.content or ""


def upload_audio_to_storage(audio_file_path: str) -> dict:
    """
    Upload audio file to Supabase Storage

    Args:
        audio_file_path: Local audio file path

    Returns:
        dict: Dictionary containing success, file_path, public_url
    """
    try:
        # Read audio file
        with open(audio_file_path, "rb") as f:
            audio_data = f.read()

        # Generate filename if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.basename(audio_file_path)
        upload_path = f"{timestamp}_{original_filename}"

        # Upload to 'audios' bucket
        response = supabase.storage.from_("audios").upload(
            path=upload_path,
            file=audio_data,
            file_options={
                "content-type": "audio/flac",
                "upsert": "true",  # Overwrite if file exists
            },
        )

        # Get public URL
        public_url = supabase.storage.from_("audios").get_public_url(upload_path)

        return {
            "success": True,
            "public_url": public_url,
            "message": "Audio uploaded successfully",
        }

    except Exception as e:
        return {"success": False, "error": str(e), "message": "Audio upload failed"}


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


def get_video_duration(video_path: Path, fallback: float = 10.0) -> float:
    try:
        with av.open(str(video_path)) as container:
            duration = None
            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if video_stream is not None:
                if video_stream.duration and video_stream.time_base:
                    duration = float(video_stream.duration * video_stream.time_base)
                elif video_stream.frames and video_stream.average_rate:
                    duration = float(video_stream.frames / video_stream.average_rate)
            if duration is None and container.duration:
                duration = float(container.duration * av.time_base)
            if duration is not None and duration > 0:
                return duration
    except Exception as exc:
        log.warning("Unable to determine video duration for %s: %s", video_path, exc)
    return fallback


@torch.inference_mode()
def run_inference(video_url: str, user_request: str, index: int):
    setup_eval_logging()

    # small_16k, small_44k, medium_44k, large_44k, large_44k_v2
    model: ModelConfig = all_model_cfg["large_44k_v2"]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    if video_url:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            response = requests.get(video_url)
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        video_path: Path = Path(tmp_file_path).expanduser()
    else:
        video_path = None

    duration: float = get_video_duration(video_path) if video_path else 10.0
    prompt: str = get_bgm_description(video_url, user_request) if video_url else ""
    output_dir: str = Path("./output").expanduser()
    seed: int = random.randint(0, 1000)
    num_steps: int = 25
    cfg_strength: float = 4.5

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        log.warning("CUDA/MPS are not available, running on CPU")
    dtype = torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(
        torch.load(model.model_path, map_location=device, weights_only=True)
    )
    log.info(f"Loaded weights from {model.model_path}")

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False,
    )
    feature_utils = feature_utils.to(device, dtype).eval()

    log.info(f"Using video {video_path}")
    video_info = load_video(video_path, duration)
    clip_frames = video_info.clip_frames
    sync_frames = video_info.sync_frames
    duration = video_info.duration_sec
    clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)

    seq_cfg.duration = duration
    net.update_seq_lengths(
        seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len
    )

    log.info(f"Prompt: {prompt}")

    audios = generate(
        clip_frames,
        sync_frames,
        [prompt],
        feature_utils=feature_utils,
        net=net,
        fm=fm,
        rng=rng,
        cfg_strength=cfg_strength,
    )
    audio = audios.float().cpu()[0]
    save_path = (
        output_dir / f"audio-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{index}.flac"
    )
    torchaudio.save(save_path, audio, seq_cfg.sampling_rate)
    log.info(f"Audio saved to {save_path}")
    log.info("Memory usage: %.2f GB", torch.cuda.max_memory_allocated() / (2**30))

    os.remove(video_path)

    return {
        "audio_path": save_path,
        "prompt": prompt,
    }


@app.get("/api/v1/ping")
async def ping() -> PingResponse:
    """Health check endpoint."""
    return PingResponse(status="ok", message="RhythmAI API is running")


async def process_single_inference(
    job_id: str, video_url: str, user_request: str, index: int
):
    """
    Process a single inference task.

    Args:
        job_id: Unique job identifier
        video_url: URL of the video to process
        user_request: User's music description request
        index: Index of this inference (0-2)

    Returns:
        dict with prompt and audio_url, or None if failed
    """
    try:
        log.info(f"Job {job_id} [#{index}]: Starting inference")

        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, run_inference, video_url, user_request, index
        )

        audio_path = result["audio_path"]
        prompt = result["prompt"]

        # Upload audio to Supabase
        log.info(f"Job {job_id} [#{index}]: Uploading audio to Supabase: {audio_path}")
        upload_result = await loop.run_in_executor(
            None, upload_audio_to_storage, audio_path
        )

        if not upload_result["success"]:
            raise Exception(
                f"Audio upload failed: {upload_result.get('error', 'Unknown error')}"
            )

        # Clean up local file after successful upload
        try:
            os.remove(audio_path)
            log.info(
                f"Job {job_id} [#{index}]: Cleaned up local audio file: {audio_path}"
            )
        except Exception as e:
            log.warning(
                f"Job {job_id} [#{index}]: Failed to clean up local file {audio_path}: {str(e)}"
            )

        log.info(f"Job {job_id} [#{index}]: Completed successfully")
        return {
            "prompt": prompt,
            "audio_url": upload_result["public_url"],
            "index": index,
        }

    except Exception as e:
        log.error(f"Job {job_id} [#{index}]: Error during audio generation: {str(e)}")
        return None


async def process_generation_job(job_id: str, video_url: str, user_request: str):
    """
    Background task to process audio generation with 3 parallel inferences.

    Args:
        job_id: Unique job identifier
        video_url: URL of the video to process
        user_request: User's music description request
    """
    try:
        # Update status to processing
        jobs_db[job_id]["status"] = JobStatusEnum.processing
        log.info(f"Job {job_id}: Starting processing for video: {video_url}")

        # Run 3 inferences in parallel
        tasks = [
            process_single_inference(job_id, video_url, user_request, 0),
            process_single_inference(job_id, video_url, user_request, 1),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Filter out None results (failed inferences)
        successful_results = [r for r in results if r is not None]

        if not successful_results:
            # All inferences failed
            jobs_db[job_id]["status"] = JobStatusEnum.failed
            jobs_db[job_id]["error"] = "All 3 inference tasks failed"
            log.error(f"Job {job_id}: All inference tasks failed")
        else:
            # At least one succeeded
            jobs_db[job_id]["status"] = JobStatusEnum.completed
            jobs_db[job_id]["results"] = successful_results
            jobs_db[job_id]["completed_count"] = len(successful_results)
            log.info(
                f"Job {job_id}: Completed with {len(successful_results)}/3 successful results"
            )

    except Exception as e:
        log.error(f"Job {job_id}: Error during audio generation: {str(e)}")
        jobs_db[job_id]["status"] = JobStatusEnum.failed
        jobs_db[job_id]["error"] = str(e)


@app.post("/api/v1/gen", response_model=GenerateResponse)
async def generate_audio(request: GenerateRequest):
    """
    Start background music generation for a video with 3 parallel inferences.

    Args:
        request: JSON body containing video_url and user_request

    Returns:
        JSON response with job_id to query for results
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Initialize job in database
        jobs_db[job_id] = {
            "status": JobStatusEnum.pending,
            "video_url": request.video_url,
            "user_request": request.user_request,
            "results": [],
            "completed_count": 0,
            "total_count": 2,
            "error": None,
        }

        log.info(f"Created job {job_id} for video: {request.video_url}")

        # Start background task with 2 parallel inferences
        asyncio.create_task(
            process_generation_job(job_id, request.video_url, request.user_request)
        )

        # Return job_id immediately
        return GenerateResponse(job_id=job_id)

    except Exception as e:
        log.error(f"Error creating generation job: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create generation job: {str(e)}"
        )


@app.get("/api/v1/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status and results of a generation job (2 parallel inferences).

    Args:
        job_id: The unique job identifier

    Returns:
        JSON response with job status and list of completed audio results
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs_db[job_id]

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        completed_count=job.get("completed_count", 0),
        total_count=job.get("total_count", 2),
        results=[AudioResult(**r) for r in job.get("results", [])],
        error=job.get("error"),
    )


if __name__ == "__main__":
    # For testing during development
    # run_inference(
    #     "https://wxoealemfuynenwgqazp.supabase.co/storage/v1/object/public/videos/videos/1763616154488-video.mp4"
    # )

    # Run the API server
    # Use 0.0.0.0 for direct access, or 127.0.0.1 if behind nginx
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
