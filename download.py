"""Modal download entry for Wan2.2-Animate.

Run:
  modal run download.py::download

Downloads the Wan2.2-Animate-14B weights (main DiT/VAE/T5 + the preprocessing
``process_checkpoint`` with pose2d/det checkpoints) to the shared ``models``
volume. The large FLUX.1-Kontext image-editing model is skipped — it is only
needed for the ``--use_flux`` retarget variant, which this plugin does not use.
"""

from __future__ import annotations

import os
from typing import Any

import modal

_cfg: dict[str, Any] = {}

WAN_REPO_ID = "Wan-AI/Wan2.2-Animate-14B"
WAN_DIR = f"/models/{WAN_REPO_ID}"

# FLUX image-editing model (~20GB+) is only used by `--use_flux` retargeting,
# which we don't enable; skip it to keep the download lean.
IGNORE_PATTERNS = ["process_checkpoint/FLUX.1-Kontext-dev/*"]

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(
        "huggingface_hub>=0.34.0,<1.0"
    ),
    volumes={"/models": volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    # Wan2.2-Animate-14B is a public repo; token is optional but harmless.
    token = os.environ.get("HF_TOKEN")

    # Always run snapshot_download — it is resumable and only fetches files that
    # are missing or changed, so it completes a previous partial download (the
    # earlier marker check wrongly skipped this when only config.json existed).
    os.makedirs(WAN_DIR, exist_ok=True)
    print(f"Downloading {WAN_REPO_ID} (excluding FLUX) ...")
    snapshot_download(
        repo_id=WAN_REPO_ID,
        local_dir=WAN_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=IGNORE_PATTERNS,
        token=token,
    )
    print(f"Done: {WAN_DIR}")

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
