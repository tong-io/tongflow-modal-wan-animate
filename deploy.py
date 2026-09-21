"""Modal deploy entry for Wan-Animate-2 (DiffSynth-Studio, distilled checkpoint).

Implements the `video-image-gen-video-move` node feature: given a character image
+ a driving video, reenact the driving motion onto the character.

Wan-Animate-2 is end-to-end — it consumes the driving video directly, so there is
no pose/face extractor stage any more (v1 needed DWPose + a ComfyUI graph). We run
`Wan-AI/Wan2.2-Animate-2-14B`'s distillation checkpoint through DiffSynth-Studio's
`WanVideoPipeline`: 10 steps, no CFG, single GPU. Upstream's own repo only ships an
8-GPU FSDP pipeline and the diffusers integration is still an unmerged PR, so
DiffSynth is the one maintained single-GPU path.

Wan-Animate-2 has no character-replacement mode, so unlike v1 this plugin no longer
serves `video-image-gen-video-mix` — tongflow-modal-scail2 covers that slot.

Deploy:          modal deploy deploy.py
Download models: modal run download.py::download   (one-time, to the volume)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import modal
from tongflow import deploy
from tongflow.models.video_image_gen_video_move import (
    VideoImageGenVideoMoveInput,
    VideoImageGenVideoMoveOutput,
)
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import current_params, node_slot


def _adv(name: str, default):
    """Advanced-section override (``TONGFLOW_SLOT_PARAMS``) or the plugin default."""
    v = current_params().get(name)
    if v is None:
        return default
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v

# Per-run knobs offered under the node's collapsed "Advanced" section.
# Pure literal (the platform scanner reads it by AST, never imports this
# module). Values reach the handlers via current_params(); an untouched
# control is absent there and falls back to the plugin default.
TONGFLOW_SLOT_PARAMS = {
    "video-image-gen-video-move": {
        "steps": {"type": "integer", "default": 10, "min": 4, "max": 40, "label": "Steps"},
        "cfg_scale": {"type": "number", "default": 1.0, "min": 1.0, "max": 6.0, "step": 0.5, "label": "CFG scale", "description": "The distilled checkpoint is trained for 1.0; raising it also needs more steps."},
        "sigma_shift": {"type": "number", "default": 5.0, "min": 1.0, "max": 10.0, "step": 0.5, "label": "Shift"},
        "fps": {"type": "integer", "default": 24, "min": 8, "max": 30, "label": "Output FPS"},
    },
}

# Slots this plugin is the default implementation of: the node picker lists
# it first and a newly added node preselects it. Read statically by the
# scanner (never executed), so any SDK version imports this file fine.
TONGFLOW_DEFAULT_SLOTS = ["video-image-gen-video-move"]

ANIMATE2_DIR = "/models/Wan-AI/Wan2.2-Animate-2-14B"
DIT = f"{ANIMATE2_DIR}/wan_animate_2/wan_animate_2_bf16_distillation.safetensors"
T5 = f"{ANIMATE2_DIR}/videomodel/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth"
CLIP = f"{ANIMATE2_DIR}/videomodel/Wan-AI/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
TOKENIZER = f"{ANIMATE2_DIR}/videomodel/Wan-AI/umt5-xxl"
VAE = "/models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"

# Distillation checkpoint constants — upstream-prescribed, not ABI knobs.
LOG_SCALE = -1.3
# Fixed context prompt for the driving video (upstream's own default).
PROMPT_REF = "视频中的人在做动作，背景静止"
# Upstream asks for a caption of the character's appearance + the background.
DEFAULT_PROMPT = "人物外观描述：画面中的人物保持参考图中的外观。 背景描述：背景保持参考图中的场景。"
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
       "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
       "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
       "杂乱的背景，三条腿，背景人很多，倒着走")

# Denoising happens per clip; longer driving videos are chained clip by clip with
# FIRST_NUM frames of overlap so identity and motion carry across the seam.
CLIP_LEN = 81  # must stay 4n+1 for the Wan VAE's temporal stride
FIRST_NUM = 1
# Output resolution follows the character image; cap the pixel budget so a large
# input can't blow up attention cost. 720x1280 is upstream's 720P setting.
MAX_AREA = 720 * 1280
# Hard cap on how much driving video we consume when the node leaves duration unset.
MAX_SECONDS = 30.0

volume = modal.Volume.from_name("models", create_if_missing=True)

APP_NAME = Path(__file__).resolve().parent.name
app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "tongflow==0.3.3",
        "fastapi[standard]",
        "diffsynth==2.1.8",
        "transformers==4.57.6",
        "imageio[ffmpeg]==2.37.2",
        "pillow==12.1.1",
    )
    # DiffSynth pulls weights from ModelScope unless told otherwise; ours are
    # already on the volume and every ModelConfig is built with an explicit
    # `path=`, so nothing should ever reach the network at boot.
    .env({"HF_HOME": "/models/hf", "DIFFSYNTH_SKIP_DOWNLOAD": "True"})
)

with image.imports():
    import imageio.v2 as imageio
    import torch
    from diffsynth.core.loader.config import ModelConfig
    from diffsynth.pipelines.wan_video import WanVideoPipeline
    from diffsynth.utils.data import save_video
    from PIL import Image


def _maybe_bytes(val: object) -> Optional[bytes]:
    if val is None:
        return None
    try:
        return prompt_media_to_bytes(val)
    except (TypeError, ValueError):
        return None


def _align16(v: float) -> int:
    """Wan's VAE downsamples 8x and the DiT patchifies 2x -> multiples of 16."""
    return max(16, int(round(v / 16.0)) * 16)


def _target_size(img_path: str, width: Optional[int], height: Optional[int]) -> tuple[int, int]:
    """Output (width, height): the node's fields when set, else the character
    image's own size, aligned to 16 and scaled down to the pixel budget."""
    w = width or 0
    h = height or 0
    if w <= 0 or h <= 0:
        with Image.open(img_path) as im:
            w, h = im.size
    area = float(w * h)
    if area > MAX_AREA:
        scale = (MAX_AREA / area) ** 0.5
        w, h = w * scale, h * scale
    return _align16(w), _align16(h)


def _letterbox(frame, width: int, height: int):
    """Fit into width x height preserving aspect, padding with black.

    The pipeline itself only does a plain `.resize()`, which would stretch a
    driving video whose aspect differs from the character image.
    """
    w, h = frame.size
    if (w, h) == (width, height):
        return frame
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(frame.resize((nw, nh), Image.LANCZOS), ((width - nw) // 2, (height - nh) // 2))
    return canvas


def _driving_frames(path: str, fps: int, seconds: float, width: int, height: int) -> list:
    """Decode the driving video, resampled to `fps` and letterboxed to the output size.

    Decoding is sequential (one pass) because random access through imageio's
    ffmpeg reader is slow; frames are emitted whenever the output clock catches up,
    which both drops frames (source faster than fps) and repeats them (slower).
    """
    limit = seconds if seconds > 0 else MAX_SECONDS
    max_frames = max(1, int(round(min(limit, MAX_SECONDS) * fps)))
    reader = imageio.get_reader(path)
    try:
        src_fps = float(reader.get_meta_data().get("fps") or fps)
    except Exception:
        src_fps = float(fps)
    if src_fps <= 0:
        src_fps = float(fps)

    frames: list = []
    next_t, step = 0.0, 1.0 / fps
    try:
        for i, raw in enumerate(reader):
            t = i / src_fps
            pil = None
            while next_t <= t + 1e-9 and len(frames) < max_frames:
                if pil is None:
                    pil = _letterbox(Image.fromarray(raw).convert("RGB"), width, height)
                frames.append(pil)
                next_t += step
            if len(frames) >= max_frames:
                break
    finally:
        reader.close()
    return frames


def _zigzag_padding(frames: list, target_len: int) -> list:
    """Extend to target_len by bouncing back and forth, as upstream's demo does."""
    if len(frames) == 1:
        return [frames[0]] * target_len
    idx, flip, out = 0, False, []
    while len(out) < target_len:
        out.append(frames[idx])
        idx += -1 if flip else 1
        if idx == 0 or idx == len(frames) - 1:
            flip = not flip
    return out[:target_len]


def _generate(pipe, reference_image, driving: list, clip_len: int, first_num: int, **kwargs) -> list:
    """Run the driving video clip by clip, carrying `first_num` frames over each seam."""
    real_len = len(driving)
    if real_len == 0:
        return []
    step = clip_len - first_num
    num_clips = 1 if real_len <= clip_len else (real_len - clip_len + step - 1) // step + 1
    target_len = clip_len + (num_clips - 1) * step
    if real_len < target_len:
        driving = _zigzag_padding(driving, target_len)

    out: list = []
    prev_tail = None
    for i in range(num_clips):
        start = i * step
        seg = pipe(
            animate2_reference_image=reference_image,
            animate2_reference_video=driving[start:start + clip_len],
            animate2_refert_images=None if i == 0 else prev_tail,
            num_frames=clip_len,
            **kwargs,
        )
        prev_tail = seg[-first_num:]
        if i != 0:
            seg = seg[first_num:]
        out.extend(seg)
    return out[:real_len]


@deploy
@app.cls(image=image, gpu="A100-80GB", volumes={"/models": volume},
         timeout=3600, scaledown_window=2)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        """Load the pipeline once; the container is reused and stays warm."""
        vram = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cuda",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(path=DIT, **vram),
                ModelConfig(path=T5, **vram),
                ModelConfig(path=CLIP, **vram),
                ModelConfig(path=VAE, **vram),
            ],
            tokenizer_config=ModelConfig(path=TOKENIZER),
        )

    @modal.method()
    @node_slot(NodeSlots.VIDEO_IMAGE_GEN_VIDEO_MOVE)
    def video_image_gen_video_move(
        self, input: VideoImageGenVideoMoveInput
    ) -> VideoImageGenVideoMoveOutput:
        img_b = _maybe_bytes(input.image)
        if not img_b:
            return VideoImageGenVideoMoveOutput(success=False, error="Missing image")
        drive_b = _maybe_bytes(input.video)
        if not drive_b:
            return VideoImageGenVideoMoveOutput(
                success=False, error="Missing reference (driving) video")

        work = "/tmp/wananimate2"
        os.makedirs(work, exist_ok=True)
        ref_path = f"{work}/ref.png"
        drive_path = f"{work}/drive.mp4"
        out_path = f"{work}/out.mp4"
        with open(ref_path, "wb") as f:
            f.write(img_b)
        with open(drive_path, "wb") as f:
            f.write(drive_b)

        fps = int(_adv("fps", 24))
        width, height = _target_size(ref_path, input.width, input.height)
        seconds = float(input.duration) if input.duration is not None else 0.0

        driving = _driving_frames(drive_path, fps, seconds, width, height)
        if not driving:
            return VideoImageGenVideoMoveOutput(
                success=False, error="Driving video has no decodable frames")

        with Image.open(ref_path) as im:
            reference = _letterbox(im.convert("RGB"), width, height)

        frames = _generate(
            self.pipe,
            reference_image=reference,
            driving=driving,
            clip_len=CLIP_LEN,
            first_num=FIRST_NUM,
            prompt=(input.text or "").strip() or DEFAULT_PROMPT,
            negative_prompt=NEG,
            animate2_prompt_ref=PROMPT_REF,
            animate2_offload_kv=True,
            animate2_log_scale=LOG_SCALE,
            height=height,
            width=width,
            num_inference_steps=_adv("steps", 10),
            cfg_scale=_adv("cfg_scale", 1.0),
            sigma_shift=_adv("sigma_shift", 5.0),
            seed=int(input.seed) if input.seed is not None else 42,
            tiled=True,
        )
        if not frames:
            return VideoImageGenVideoMoveOutput(success=False, error="No frames generated")

        save_video(frames, out_path, fps=fps, quality=5)
        with open(out_path, "rb") as f:
            raw = f.read()
        if not raw:
            return VideoImageGenVideoMoveOutput(success=False, error="Empty video output")
        return VideoImageGenVideoMoveOutput(success=True, video=asset(raw, mime="video/mp4"))

    # Cloud single-node self-serve: ONE container, direct browser stream. The
    # browser's EventSource is 302'd here with taskId/token/origin; serve_stream
    # _from_spec (SDK) fetches the run spec from the Worker, runs the slot
    # in-container, and streams progress + result. Streaming dodges the 150s
    # cap. Label is uniform (`<app>-serve`) so the Worker derives the URL.
    @modal.fastapi_endpoint(method="GET", label=f"{APP_NAME}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )
