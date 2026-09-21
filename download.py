"""Modal download entry for Wan-Animate-2.

Run:
  modal run download.py::download

Fetches, into the shared ``models`` volume, exactly what deploy.py loads:
the Animate-2 distillation DiT plus the UMT5 text encoder, CLIP vision encoder
and UMT5 tokenizer that ship alongside it, and the Wan2.1 VAE (which lives in
the Wan2.1-T2V-14B repo, matching DiffSynth-Studio's own Animate-2 example).

The base (non-distilled) checkpoint and the FLUX retarget model are skipped —
the plugin runs the 10-step distillation path only.

Both repos are public and Apache 2.0; ``HF_TOKEN`` is optional and only helps
with Hugging Face rate limits.
"""

from __future__ import annotations

import os

import modal

ANIMATE2_REPO = "Wan-AI/Wan2.2-Animate-2-14B"
VAE_REPO = "Wan-AI/Wan2.1-T2V-14B"

ANIMATE2_DIR = f"/models/{ANIMATE2_REPO}"
VAE_DIR = f"/models/{VAE_REPO}"

# Everything deploy.py opens by path, and nothing else.
ANIMATE2_PATTERNS = [
    "wan_animate_2/wan_animate_2_bf16_distillation.safetensors",
    "videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth",
    "videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "videomodel/Wan-AI/umt5-xxl/*",
]
VAE_PATTERNS = ["Wan2.1_VAE.pth"]

volume = modal.Volume.from_name("models", create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
        "huggingface_hub>=0.34.0,<1.0"
    ),
    volumes={"/models": volume},
    timeout=7200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    # snapshot_download is resumable and only fetches what is missing or changed,
    # so it also completes an earlier partial download — never skip it on a
    # directory that merely exists.
    for repo_id, local_dir, patterns in (
        (ANIMATE2_REPO, ANIMATE2_DIR, ANIMATE2_PATTERNS),
        (VAE_REPO, VAE_DIR, VAE_PATTERNS),
    ):
        os.makedirs(local_dir, exist_ok=True)
        print(f"Downloading {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=patterns,
            token=token,
        )
        print(f"Done: {local_dir}")

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
