"""Modal deploy entry for Wan2.2-Animate (ComfyUI + Kijai fp8).

Implements the `video-image-gen-video-move` node feature: given a character image
+ a reference (driving) video, reenact the driving motion onto the character.

Runs headless ComfyUI + ComfyUI-WanVideoWrapper with Kijai's fp8 Wan2.2-Animate-14B
plus the lightx2v 6-step distill LoRA — ~10-30x faster and ~22GB VRAM vs the official
bf16 generate.py path (which timed out). The ComfyUI server boots once per container
(@modal.enter) and is reused; models stay resident in VRAM across calls.

Deploy:        modal deploy deploy.py
Download models: modal run comfy_app.py::download_models   (one-time, to the volume)
"""

from __future__ import annotations
from pathlib import Path

import os
from typing import Any, Optional

import modal
from tongflow import deploy
from tongflow.models.video_image_gen_video_mix import (
    VideoImageGenVideoMixInput,
    VideoImageGenVideoMixOutput,
)
from tongflow.models.video_image_gen_video_move import (
    VideoImageGenVideoMoveInput,
    VideoImageGenVideoMoveOutput,
)
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot

COMFY = "/opt/ComfyUI"
COMFY_MODELS = "/models/comfyui"

volume = modal.Volume.from_name("models", create_if_missing=True)

# Model filenames (flat, as downloaded by comfy_app.py::download_models).
DIT = "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
T5 = "umt5-xxl-enc-bf16.safetensors"
CLIPV = "clip_vision_h.safetensors"
LORA_RELIGHT = "WanAnimate_relight_lora_fp16.safetensors"
LORA_LIGHTX2V = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
SAM3 = "sam3-fp16.safetensors"  # person segmentation for replace mode

DEFAULT_PROMPT = "high quality video, natural motion, consistent character"
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
       "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
       "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
       "杂乱的背景，三条腿，背景人很多，倒着走")

# Custom node packs the WanAnimate graph needs.
CUSTOM_NODES = {
    "ComfyUI-WanVideoWrapper": "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
    "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
    "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    "comfyui_controlnet_aux": "https://github.com/Fannovel16/comfyui_controlnet_aux.git",
    "ComfyUI-segment-anything-2": "https://github.com/kijai/ComfyUI-segment-anything-2.git",
    # SAM3: text-promptable person segmentation + tracking for replace mode.
    "ComfyUI-Easy-Sam3": "https://github.com/yolain/ComfyUI-Easy-Sam3.git",
}
_clone_cmds = []
for _name, _url in CUSTOM_NODES.items():
    _dst = f"{COMFY}/custom_nodes/{_name}"
    _clone_cmds.append(f"git clone --depth 1 {_url} {_dst}")
    _clone_cmds.append(
        f"[ -f {_dst}/requirements.txt ] && pip install -r {_dst}/requirements.txt || true"
    )

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "ffmpeg", "build-essential")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git " + COMFY,
        f"pip install -r {COMFY}/requirements.txt",
        *_clone_cmds,
    )
    .pip_install("tongflow==0.1.0")
    .env({"PYTHONPATH": COMFY, "HF_HOME": "/models/hf"})
)

with image.imports():
    import json
    import subprocess
    import time
    import urllib.error
    import urllib.request


def _maybe_bytes(val: object) -> Optional[bytes]:
    if val is None:
        return None
    try:
        return prompt_media_to_bytes(val)
    except (TypeError, ValueError):
        return None


_FPS = 16


def _probe(img_path: str, vid_path: str, duration: object) -> tuple[int, int, int]:
    """Derive output (width, height, frame_cap) from the inputs.

    - width/height: the character image's NATIVE resolution (no scaling), only
      aligned to multiples of 16 (Wan VAE x8 * patch x2). Aspect ratio comes from
      the image, never from a node field.
    - frame_cap: VHS_LoadVideo `frame_load_cap` = duration (seconds) * 16 fps. This
      ONLY caps how many frames load; the actual num_frames fed to the model comes
      from VHS's own frame_count output (see _build_workflow), so num_frames always
      equals the real pose frames and the tail can never reflect-pad into reverse.
    """
    import cv2

    im = cv2.imread(img_path)
    h, w = (im.shape[0], im.shape[1]) if im is not None else (832, 480)
    W = max(16, int(round(w / 16.0)) * 16)
    H = max(16, int(round(h / 16.0)) * 16)

    try:
        secs = float(duration) if duration is not None else 0.0
    except (TypeError, ValueError):
        secs = 0.0
    frame_cap = int(round(secs * _FPS)) if secs > 0 else 161
    frame_cap = max(13, min(frame_cap, 401))
    return W, H, frame_cap


def _build_workflow(img_name, vid_name, prompt, width, height, frame_cap, seed):
    """Minimal pose-driven Wan2.2-Animate graph: fp8 DiT + relight + lightx2v 6-step.

    num_frames is wired from VHS_LoadVideo's frame_count output (out 1) — exactly
    what the official workflow does — so it always matches the real pose frames.
    """
    return {
        "load_img": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "load_vid": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": vid_name, "force_rate": 16, "custom_width": 0, "custom_height": 0,
            "frame_load_cap": frame_cap, "skip_first_frames": 0, "select_every_nth": 1}},
        "dwpose": {"class_type": "DWPreprocessor", "inputs": {
            "image": ["load_vid", 0], "detect_hand": "enable", "detect_body": "enable",
            "detect_face": "enable", "resolution": 832,
            "bbox_detector": "yolox_l.torchscript.pt",
            "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
            "scale_stick_for_xinsr_cn": "disable"}},
        "clip_loader": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIPV}},
        "clip_enc": {"class_type": "WanVideoClipVisionEncode", "inputs": {
            "clip_vision": ["clip_loader", 0], "image_1": ["load_img", 0],
            "strength_1": 1.0, "strength_2": 1.0, "crop": "center",
            "combine_embeds": "average", "force_offload": True}},
        "vae": {"class_type": "WanVideoVAELoader", "inputs": {
            "model_name": VAE, "precision": "bf16"}},
        "lora": {"class_type": "WanVideoLoraSelectMulti", "inputs": {
            "lora_0": LORA_RELIGHT, "strength_0": 1.0,
            "lora_1": LORA_LIGHTX2V, "strength_1": 1.0,
            "lora_2": "none", "strength_2": 1.0, "lora_3": "none", "strength_3": 1.0,
            "lora_4": "none", "strength_4": 1.0, "merge_loras": False}},
        "model": {"class_type": "WanVideoModelLoader", "inputs": {
            "model": DIT, "base_precision": "fp16_fast",
            "quantization": "fp8_e4m3fn_scaled", "load_device": "offload_device",
            "attention_mode": "sdpa", "lora": ["lora", 0]}},
        "text": {"class_type": "WanVideoTextEncodeCached", "inputs": {
            "model_name": T5, "precision": "bf16", "positive_prompt": prompt,
            "negative_prompt": NEG, "quantization": "disabled",
            "use_disk_cache": False, "device": "gpu"}},
        "embeds": {"class_type": "WanVideoAnimateEmbeds", "inputs": {
            "vae": ["vae", 0], "width": width, "height": height,
            "num_frames": ["load_vid", 1], "force_offload": True,
            "frame_window_size": 77, "colormatch": "disabled",
            "pose_strength": 1.0, "face_strength": 1.0,
            "clip_embeds": ["clip_enc", 0], "ref_images": ["load_img", 0],
            "pose_images": ["dwpose", 0]}},
        "sampler": {"class_type": "WanVideoSampler", "inputs": {
            "model": ["model", 0], "image_embeds": ["embeds", 0], "steps": 6,
            "cfg": 1.0, "shift": 5.0, "seed": seed, "force_offload": True,
            "scheduler": "dpm++_sde", "riflex_freq_index": 0, "text_embeds": ["text", 0]}},
        "decode": {"class_type": "WanVideoDecode", "inputs": {
            "vae": ["vae", 0], "samples": ["sampler", 0], "enable_vae_tiling": False,
            "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
        # WanAnim looping rounds up to whole frame_window_size windows and reflect-pads
        # the pose tail (reversed tail). Trim back to the real loaded frame count.
        "trim": {"class_type": "GetImageRangeFromBatch", "inputs": {
            "images": ["decode", 0], "start_index": 0, "num_frames": ["load_vid", 1]}},
        "save": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["trim", 0], "frame_rate": 16, "loop_count": 0,
            "filename_prefix": "wananim", "format": "video/h264-mp4",
            "pingpong": False, "save_output": True}},
    }


def _probe_video(vid_path: str) -> tuple[int, int, int]:
    """Replace mode: output resolution = driving video's (aligned 16). frame_cap=0
    means VHS loads the WHOLE video so the result matches the reference length —
    WanAnim looping keeps VRAM bounded per window regardless of total length."""
    import cv2

    cap = cv2.VideoCapture(vid_path)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 832)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    cap.release()
    W = max(16, int(round(vw / 16.0)) * 16)
    H = max(16, int(round(vh / 16.0)) * 16)
    return W, H, 0


def _build_replace_workflow(img_name, vid_name, prompt, width, height, frame_cap, seed):
    """Replace mode: SAM3 auto-segments the person in the driving video; the masked
    region is inpainted with the character (pose-driven), keeping the original scene."""
    wf = _build_workflow(img_name, vid_name, prompt, width, height, frame_cap, seed)
    wf["load_vid"]["inputs"]["custom_width"] = width
    wf["load_vid"]["inputs"]["custom_height"] = height
    wf["sam3_model"] = {"class_type": "easy sam3ModelLoader", "inputs": {
        "model": SAM3, "segmentor": "video", "device": "cuda", "precision": "fp16"}}
    wf["sam3_seg"] = {"class_type": "easy sam3VideoSegmentation", "inputs": {
        "sam3_model": ["sam3_model", 0], "video_frames": ["load_vid", 0],
        "prompt": "person", "frame_index": 0, "object_id": 1,
        "score_threshold_detection": 0.5, "new_det_thresh": 0.7,
        "propagation_direction": "both", "start_frame_index": 0,
        "max_frames_to_track": -1, "close_after_propagation": True,
        "keep_model_loaded": False}}
    # Official mask pipeline: SAM -> GrowMask -> BlockifyMask. Blockify coarsens the
    # tight silhouette into blocks so a differently-shaped character isn't clipped to
    # the original person's outline (which produced black edges).
    wf["grow"] = {"class_type": "GrowMask", "inputs": {
        "mask": ["sam3_seg", 0], "expand": 10, "tapered_corners": True}}
    wf["blockify"] = {"class_type": "BlockifyMask", "inputs": {
        "masks": ["grow", 0], "block_size": 32, "device": "gpu"}}
    # Black out the masked region in bg so the original person doesn't leak through
    # bg conditioning — otherwise the reference character never replaces them.
    wf["bg_masked"] = {"class_type": "DrawMaskOnImage", "inputs": {
        "image": ["load_vid", 0], "mask": ["blockify", 0], "color": "0, 0, 0",
        "device": "gpu"}}
    wf["embeds"]["inputs"]["bg_images"] = ["bg_masked", 0]
    wf["embeds"]["inputs"]["mask"] = ["blockify", 0]
    wf["save"]["inputs"]["filename_prefix"] = "wanrepl"
    return wf


def _submit_graph(base, wf):
    """Submit a ComfyUI workflow, poll, return (True, mp4_bytes) or (False, error)."""
    body = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(f"{base}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        return False, f"workflow rejected: {e.read().decode()[:1000]}"
    out = None
    for _ in range(1800):
        time.sleep(1)
        with urllib.request.urlopen(f"{base}/history/{pid}", timeout=10) as r:
            hist = json.loads(r.read())
        if pid in hist:
            h = hist[pid]
            if h.get("outputs"):
                out = h["outputs"]
                break
            if h.get("status", {}).get("status_str") == "error":
                return False, "comfy error: " + json.dumps(h.get("status"))[:1200]
    if not out:
        return False, "timed out"
    for node_out in out.values():
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key, []):
                fn, sub = item.get("filename"), item.get("subfolder", "")
                typ = item.get("type", "output")
                d = {"output": "output", "temp": "temp"}.get(typ, "output")
                path = os.path.join(COMFY, d, sub, fn or "")
                if fn and fn.endswith((".mp4", ".webm")) and os.path.isfile(path):
                    with open(path, "rb") as fh:
                        raw = fh.read()
                    if raw:
                        return True, raw
    return False, "no video output"


@deploy
@app.cls(image=image, gpu="A100-80GB", volumes={"/models": volume},
         timeout=1800, scaledown_window=5)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        """Boot the ComfyUI server once; reused across calls (models stay warm)."""
        os.makedirs(COMFY_MODELS, exist_ok=True)
        with open(os.path.join(COMFY, "extra_model_paths.yaml"), "w") as f:
            f.write(
                "wan_volume:\n"
                f"  base_path: {COMFY_MODELS}/\n"
                "  diffusion_models: diffusion_models\n  vae: vae\n"
                "  text_encoders: text_encoders\n  clip_vision: clip_vision\n"
                "  loras: loras\n"
            )
        # sam3 is a custom model-folder type — symlink it so LoadSam3Model finds it.
        os.makedirs(f"{COMFY_MODELS}/sam3", exist_ok=True)
        link = f"{COMFY}/models/sam3"
        if not os.path.islink(link) and not os.path.isdir(link):
            os.makedirs(f"{COMFY}/models", exist_ok=True)
            try:
                os.symlink(f"{COMFY_MODELS}/sam3", link)
            except FileExistsError:
                pass
        self.proc = subprocess.Popen(
            ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
             "--disable-auto-launch"],
            cwd=COMFY,
        )
        self.base = "http://127.0.0.1:8188"
        for _ in range(300):
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited early: {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.base}/object_info", timeout=2) as r:
                    if r.status == 200:
                        json.loads(r.read())
                        return
            except Exception:
                time.sleep(1)
        raise RuntimeError("ComfyUI server did not become ready")

    @modal.exit()
    def _shutdown(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass

    @modal.method()
    @node_slot(NodeSlots.VIDEO_IMAGE_GEN_VIDEO_MOVE)
    def video_image_gen_video_move(
        self, input: VideoImageGenVideoMoveInput
    ) -> VideoImageGenVideoMoveOutput:
        img_b = _maybe_bytes(input.image)
        if not img_b:
            return VideoImageGenVideoMoveOutput(success=False, error="Missing image")
        ref_b = _maybe_bytes(input.video)
        if not ref_b:
            return VideoImageGenVideoMoveOutput(
                success=False, error="Missing reference (driving) video")

        prompt = (input.text or "").strip() or DEFAULT_PROMPT
        seed = int(input.seed) if input.seed is not None else 42

        os.makedirs(f"{COMFY}/input", exist_ok=True)
        ref_path = f"{COMFY}/input/ref.png"
        drive_path = f"{COMFY}/input/drive.mp4"
        with open(ref_path, "wb") as f:
            f.write(img_b)
        with open(drive_path, "wb") as f:
            f.write(ref_b)

        # Resolution follows the character image; duration only caps how many frames
        # VHS loads (frame_cap). num_frames is taken from VHS's real frame_count
        # inside the graph, so the motion never reflect-pads into reverse.
        width, height, frame_cap = _probe(ref_path, drive_path, input.duration)

        wf = _build_workflow("ref.png", "drive.mp4", prompt, width, height,
                             frame_cap, seed)
        ok, res = _submit_graph(self.base, wf)
        if ok:
            return VideoImageGenVideoMoveOutput(
                success=True, video=asset(res, mime="video/mp4"))
        return VideoImageGenVideoMoveOutput(success=False, error=str(res))

    @modal.method()
    @node_slot(NodeSlots.VIDEO_IMAGE_GEN_VIDEO_MIX)
    def video_image_gen_video_mix(
        self, input: VideoImageGenVideoMixInput
    ) -> VideoImageGenVideoMixOutput:
        """Replace mode: swap the person in the driving video with the character,
        keeping the original scene (SAM3 auto-segments the person)."""
        img_b = _maybe_bytes(input.image)
        if not img_b:
            return VideoImageGenVideoMixOutput(success=False, error="Missing image")
        ref_b = _maybe_bytes(input.video)
        if not ref_b:
            return VideoImageGenVideoMixOutput(
                success=False, error="Missing reference (driving) video")

        prompt = (input.text or "").strip() or DEFAULT_PROMPT

        os.makedirs(f"{COMFY}/input", exist_ok=True)
        ref_path = f"{COMFY}/input/ref.png"
        drive_path = f"{COMFY}/input/drive.mp4"
        with open(ref_path, "wb") as f:
            f.write(img_b)
        with open(drive_path, "wb") as f:
            f.write(ref_b)

        # Replace output keeps the driving video's resolution.
        width, height, frame_cap = _probe_video(drive_path)
        wf = _build_replace_workflow("ref.png", "drive.mp4", prompt, width, height,
                                     frame_cap, 42)
        ok, res = _submit_graph(self.base, wf)
        if ok:
            return VideoImageGenVideoMixOutput(
                success=True, video=asset(res, mime="video/mp4"))
        return VideoImageGenVideoMixOutput(success=False, error=str(res))
