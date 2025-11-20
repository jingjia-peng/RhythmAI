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

load_dotenv()

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


if __name__ == "__main__":
    run_inference("https://wxoealemfuynenwgqazp.supabase.co/storage/v1/object/public/videos/videos/1763616154488-video.mp4")
