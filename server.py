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
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from supabase import create_client, Client

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
Provide a detailed description of the background music the video should feature. The musical description should cover the following five elements:

1.  **Genre and Style:** Suggest a specific genre (e.g., Lo-Fi, Cinematic Orchestral, Acoustic Folk, Synthwave).
2.  **Instrumentation:** Specify the core instruments (e.g., "Main melody on piano with a cello backing," "Simple drum loop with ambient synth pads," "Acoustic guitar and light percussion").
3.  **Tempo and Dynamics:** Describe the speed (BPM) and loudness changes (e.g., "Medium tempo, 100 BPM, starting softly and building dynamics at the 0:45 mark.").
4.  **Mood and Key:** State the emotional effect and suggest a musical key (e.g., "The mood should be hopeful and slightly melancholic, suggesting a minor key like C minor.").
5.  **Placement Notes:** Provide guidance on where the music should be most prominent or subtle (e.g. "Use a climactic chord progression for the final 15 seconds.").

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


class PingResponse(BaseModel):
    status: str
    message: str


class GenerateResponse(BaseModel):
    prompt: str
    audio_url: str


def get_bgm_description(video_url: str) -> str:
    completion = client.chat.completions.create(
        model="ernie-4.5-vl-28b-a3b",
        temperature=0.6,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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


@torch.inference_mode()
def run_inference(video_url: str):
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

    prompt: str = get_bgm_description(video_url) if video_url else ""
    output_dir: str = Path("./output").expanduser()
    seed: int = 42
    num_steps: int = 25
    duration: float = 8.0  # TODO: chagne to input video duration
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
    save_path = output_dir / f"audio-{datetime.now().strftime('%Y%m%d_%H%M%S')}.flac"
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


@app.post("/api/v1/gen", response_model=GenerateResponse)
async def generate_audio(request: GenerateRequest):
    """
    Generate background music for a video.

    Args:
        request: JSON body containing video_url

    Returns:
        JSON response with prompt and audio file URL from Supabase
    """
    try:
        log.info(f"Received generation request for video: {request.video_url}")
        result = run_inference(request.video_url)

        audio_path = result["audio_path"]
        prompt = result["prompt"]

        # Upload audio to Supabase
        log.info(f"Uploading audio to Supabase: {audio_path}")
        upload_result = upload_audio_to_storage(audio_path)

        if not upload_result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Audio upload failed: {upload_result.get('error', 'Unknown error')}",
            )

        # Clean up local file after successful upload
        try:
            os.remove(audio_path)
            log.info(f"Cleaned up local audio file: {audio_path}")
        except Exception as e:
            log.warning(f"Failed to clean up local file {audio_path}: {str(e)}")

        # Return JSON response with prompt and URL
        return GenerateResponse(
            prompt=prompt,
            audio_url=upload_result["public_url"],
        )

    except Exception as e:
        log.error(f"Error during audio generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Audio generation failed: {str(e)}"
        )


if __name__ == "__main__":
    # For testing during development
    # run_inference(
    #     "https://wxoealemfuynenwgqazp.supabase.co/storage/v1/object/public/videos/videos/1763616154488-video.mp4"
    # )

    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
