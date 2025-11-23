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


class MergeRequest(BaseModel):
    video_url: str
    audio_url: str


class PingResponse(BaseModel):
    status: str
    message: str


class GenerateResponse(BaseModel):
    job_id: str


class MergeResponse(BaseModel):
    video_url: str


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

# Maximum number of jobs to keep in memory (to prevent memory growth)
MAX_JOBS_IN_MEMORY = 100


def cleanup_old_jobs():
    """Remove old completed/failed jobs if we exceed MAX_JOBS_IN_MEMORY."""
    if len(jobs_db) > MAX_JOBS_IN_MEMORY:
        # Sort by creation time (job_id is uuid, so we can't sort by that)
        # Instead, keep only the most recent MAX_JOBS_IN_MEMORY jobs
        completed_or_failed = [
            job_id
            for job_id, job in jobs_db.items()
            if job["status"] in [JobStatusEnum.completed, JobStatusEnum.failed]
        ]
        # Remove oldest jobs
        jobs_to_remove = len(completed_or_failed) - (MAX_JOBS_IN_MEMORY // 2)
        if jobs_to_remove > 0:
            for job_id in completed_or_failed[:jobs_to_remove]:
                del jobs_db[job_id]
                log.info(f"Cleaned up old job {job_id} from memory")


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


def upload_video_to_storage(video_file_path: str) -> dict:
    """
    Upload video file to Supabase Storage

    Args:
        video_file_path: Local video file path

    Returns:
        dict: Dictionary containing success, file_path, public_url
    """
    try:
        # Read video file
        with open(video_file_path, "rb") as f:
            video_data = f.read()

        original_filename = os.path.basename(video_file_path)
        upload_path = f"merged_{original_filename}"

        # Upload to 'videos' bucket
        response = supabase.storage.from_("videos").upload(
            path=upload_path,
            file=video_data,
            file_options={
                "content-type": "video/mp4",
                "upsert": "true",  # Overwrite if file exists
            },
        )

        # Get public URL
        public_url = supabase.storage.from_("videos").get_public_url(upload_path)

        return {
            "success": True,
            "public_url": public_url,
            "message": "Video uploaded successfully",
        }

    except Exception as e:
        return {"success": False, "error": str(e), "message": "Video upload failed"}


def merge_video_audio(video_url: str, audio_url: str) -> str:
    """
    Merge video and audio files using ffmpeg

    Args:
        video_url: URL of the video file
        audio_url: URL of the audio file

    Returns:
        str: Path to the merged video file
    """
    import subprocess

    # Download video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        response = requests.get(video_url)
        tmp_video.write(response.content)
        video_path = tmp_video.name

    # Download audio
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp_audio:
        response = requests.get(audio_url)
        tmp_audio.write(response.content)
        audio_path = tmp_audio.name

    # Create output path
    output_path = tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False, dir=Path("./output")
    ).name

    try:
        # Use ffmpeg to merge video and audio
        # -i: input files
        # -c:v copy: copy video codec (no re-encoding)
        # -c:a aac: encode audio to AAC
        # -map 0:v:0: use video from first input
        # -map 1:a:0: use audio from second input
        # -shortest: finish encoding when shortest input ends
        # -y: overwrite output file if exists
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                "-y",
                output_path,
            ],
            check=True,
            capture_output=True,
        )

        log.info(f"Successfully merged video and audio to {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        log.error(f"FFmpeg error: {e.stderr.decode()}")
        raise Exception(f"Failed to merge video and audio: {e.stderr.decode()}")
    finally:
        # Clean up temporary files
        try:
            os.remove(video_path)
            os.remove(audio_path)
        except Exception as e:
            log.warning(f"Failed to clean up temporary files: {str(e)}")


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

# Global model instances (load once, reuse for all requests)
_model_lock = asyncio.Lock()
_models_loaded = False
_net = None
_feature_utils = None
_fm = None
_seq_cfg = None
_model_device = None
_model_dtype = None


async def load_models_once():
    """Load models once and reuse them for all requests."""
    global _models_loaded, _net, _feature_utils, _fm, _seq_cfg, _model_device, _model_dtype

    async with _model_lock:
        if _models_loaded:
            log.info("Models already loaded, skipping initialization")
            return

        log.info("Loading models for the first time...")
        setup_eval_logging()

        # Model configuration
        model: ModelConfig = all_model_cfg["large_44k_v2"]
        model.download_if_needed()
        _seq_cfg = model.seq_cfg

        # Device setup
        if torch.cuda.is_available():
            _model_device = "cuda"
        elif torch.backends.mps.is_available():
            _model_device = "mps"
        else:
            _model_device = "cpu"
            log.warning("CUDA/MPS are not available, running on CPU")
        _model_dtype = torch.bfloat16

        # Load MMAudio model
        _net = get_my_mmaudio(model.model_name).to(_model_device, _model_dtype).eval()
        _net.load_weights(
            torch.load(model.model_path, map_location=_model_device, weights_only=True)
        )
        log.info(f"Loaded MMAudio weights from {model.model_path}")

        # Load feature extraction models
        _feature_utils = FeaturesUtils(
            tod_vae_ckpt=model.vae_path,
            synchformer_ckpt=model.synchformer_ckpt,
            enable_conditions=True,
            mode=model.mode,
            bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
            need_vae_encoder=False,
        )
        _feature_utils = _feature_utils.to(_model_device, _model_dtype).eval()
        log.info("Loaded FeaturesUtils (CLIP, Synchformer, VAE, vocoder)")

        # Initialize flow matching
        _fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=25)

        _models_loaded = True
        log.info(f"All models loaded successfully on {_model_device}")

        # Clear any initialization memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info(
                f"Initial memory usage: {torch.cuda.memory_allocated() / (2**30):.2f} GB"
            )


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
    """
    Run inference using pre-loaded global models.

    This function now reuses global model instances instead of loading new ones,
    which prevents OOM issues from multiple model loads.
    """
    global _net, _feature_utils, _fm, _seq_cfg, _model_device, _model_dtype

    if not _models_loaded:
        raise RuntimeError("Models not loaded. Call load_models_once() first.")

    # Download and process video
    if video_url:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            response = requests.get(video_url)
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        video_path: Path = Path(tmp_file_path).expanduser()
    else:
        video_path = None

    try:
        duration: float = get_video_duration(video_path) if video_path else 10.0
        prompt: str = get_bgm_description(video_url, user_request) if video_url else ""
        output_dir: Path = Path("./output").expanduser()
        seed: int = random.randint(0, 1000)
        cfg_strength: float = 4.5

        output_dir.mkdir(parents=True, exist_ok=True)

        # Use pre-loaded models
        rng = torch.Generator(device=_model_device)
        rng.manual_seed(seed)

        log.info(f"Using video {video_path}")
        video_info = load_video(video_path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        duration = video_info.duration_sec
        clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)

        # Update sequence lengths for this specific video duration
        _seq_cfg.duration = duration
        _net.update_seq_lengths(
            _seq_cfg.latent_seq_len, _seq_cfg.clip_seq_len, _seq_cfg.sync_seq_len
        )

        log.info(f"Prompt: {prompt}")

        # Generate audio using pre-loaded models
        audios = generate(
            clip_frames,
            sync_frames,
            [prompt],
            feature_utils=_feature_utils,
            net=_net,
            fm=_fm,
            rng=rng,
            cfg_strength=cfg_strength,
        )
        audio = audios.float().cpu()[0]
        save_path = (
            output_dir
            / f"audio-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{index}.flac"
        )
        torchaudio.save(save_path, audio, _seq_cfg.sampling_rate)
        log.info(f"Audio saved to {save_path}")

        # Clean up GPU memory after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("Memory usage: %.2f GB", torch.cuda.memory_allocated() / (2**30))

        return {
            "audio_path": save_path,
            "prompt": prompt,
        }

    finally:
        # Always clean up temporary video file
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                log.warning(f"Failed to remove temporary video file {video_path}: {e}")


@app.on_event("startup")
async def startup_event():
    """Load models when the server starts."""
    log.info("Server starting up, loading models...")
    await load_models_once()
    log.info("Server startup complete, models ready")


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
    Background task to process audio generation with 2 sequential inferences.

    Changed from parallel to sequential execution to prevent OOM from loading
    multiple model instances simultaneously.

    Args:
        job_id: Unique job identifier
        video_url: URL of the video to process
        user_request: User's music description request
    """
    try:
        # Update status to processing
        jobs_db[job_id]["status"] = JobStatusEnum.processing
        log.info(f"Job {job_id}: Starting processing for video: {video_url}")

        # Run 2 inferences sequentially (not in parallel) to avoid OOM
        # This reuses the same model for both inferences
        successful_results = []
        for i in range(2):
            result = await process_single_inference(job_id, video_url, user_request, i)
            if result is not None:
                successful_results.append(result)
            else:
                log.warning(f"Job {job_id}: Inference #{i} failed")

        if not successful_results:
            # All inferences failed
            jobs_db[job_id]["status"] = JobStatusEnum.failed
            jobs_db[job_id]["error"] = "All 2 inference tasks failed"
            log.error(f"Job {job_id}: All inference tasks failed")
        else:
            # At least one succeeded
            jobs_db[job_id]["status"] = JobStatusEnum.completed
            jobs_db[job_id]["results"] = successful_results
            jobs_db[job_id]["completed_count"] = len(successful_results)
            log.info(
                f"Job {job_id}: Completed with {len(successful_results)}/2 successful results"
            )

    except Exception as e:
        log.error(f"Job {job_id}: Error during audio generation: {str(e)}")
        jobs_db[job_id]["status"] = JobStatusEnum.failed
        jobs_db[job_id]["error"] = str(e)


@app.post("/api/v1/gen", response_model=GenerateResponse)
async def generate_audio(request: GenerateRequest):
    """
    Start background music generation for a video with 2 sequential inferences.

    Args:
        request: JSON body containing video_url and user_request

    Returns:
        JSON response with job_id to query for results
    """
    try:
        # Clean up old jobs to prevent memory growth
        cleanup_old_jobs()

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

        # Start background task with 2 sequential inferences
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
    Get the status and results of a generation job (2 sequential inferences).

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


@app.post("/api/v1/merge", response_model=MergeResponse)
async def merge_video_and_audio(request: MergeRequest):
    """
    Merge video and audio files, then upload to Supabase.

    Args:
        request: JSON body containing video_url and audio_url

    Returns:
        JSON response with the URL of the merged video
    """
    try:
        log.info(f"Merging video {request.video_url} with audio {request.audio_url}")

        # Run merge in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        merged_video_path = await loop.run_in_executor(
            None, merge_video_audio, request.video_url, request.audio_url
        )

        # Upload merged video to Supabase
        log.info(f"Uploading merged video to Supabase: {merged_video_path}")
        upload_result = await loop.run_in_executor(
            None, upload_video_to_storage, merged_video_path
        )

        if not upload_result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Video upload failed: {upload_result.get('error', 'Unknown error')}",
            )

        # Clean up local file after successful upload
        try:
            os.remove(merged_video_path)
            log.info(f"Cleaned up local merged video file: {merged_video_path}")
        except Exception as e:
            log.warning(f"Failed to clean up local file {merged_video_path}: {str(e)}")

        log.info(
            f"Successfully merged and uploaded video: {upload_result['public_url']}"
        )
        return MergeResponse(video_url=upload_result["public_url"])

    except Exception as e:
        log.error(f"Error during video/audio merge: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video merge failed: {str(e)}")


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
