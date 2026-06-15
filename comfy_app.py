"""Standalone dev app: headless ComfyUI + WanVideoWrapper for Wan2.2-Animate fp8.

Built separately from deploy.py so we can iterate on the ComfyUI stack end-to-end
(boot -> nodes load -> models load -> animate workflow) before wiring it into the
plugin's node_slot. Milestones:
  1. image boots, custom nodes import cleanly         (modal run comfy_app.py::list_nodes)
  2. models present + model loads
  3. animate workflow runs end to end -> mp4
  4. integrate into deploy.py

Speed story vs official bf16: fp8 DiT + lightx2v 6-step distill LoRA + 832x480.
"""

from __future__ import annotations

import modal

COMFY = "/opt/ComfyUI"

# Custom node packs the WanAnimate workflow needs (class_type -> repo):
#   WanVideo* -> WanVideoWrapper, ImageResizeKJv2/PointsEditor/GrowMask -> KJNodes,
#   VHS_*    -> VideoHelperSuite, DWPreprocessor -> controlnet_aux,
#   Sam2*    -> segment-anything-2.
CUSTOM_NODES = {
    "ComfyUI-WanVideoWrapper": "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
    "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
    "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    "comfyui_controlnet_aux": "https://github.com/Fannovel16/comfyui_controlnet_aux.git",
    "ComfyUI-segment-anything-2": "https://github.com/kijai/ComfyUI-segment-anything-2.git",
    # SAM3: text-promptable concept segmentation + video tracking -> headless
    # person mask for the replace mode (no interactive PointsEditor needed).
    "ComfyUI-Easy-Sam3": "https://github.com/yolain/ComfyUI-Easy-Sam3.git",
}

_clone_cmds = []
for name, url in CUSTOM_NODES.items():
    dst = f"{COMFY}/custom_nodes/{name}"
    _clone_cmds.append(f"git clone --depth 1 {url} {dst}")
    # Install each node pack's own requirements (ignore failure to not block boot).
    _clone_cmds.append(
        f"[ -f {dst}/requirements.txt ] && pip install -r {dst}/requirements.txt || true"
    )

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "ffmpeg", "build-essential")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git " + COMFY,
        f"pip install -r {COMFY}/requirements.txt",
        *_clone_cmds,
    )
    .env({"PYTHONPATH": COMFY, "HF_HOME": "/models/hf"})
)

app = modal.App("wan-animate-comfy-dev")

# Persist big models on the shared volume, laid out as ComfyUI model dirs.
volume = modal.Volume.from_name("models", create_if_missing=True)
COMFY_MODELS = "/models/comfyui"

# (repo_id, path-in-repo, comfyui-subdir, flat-dest-name)
MODELS = [
    ("Kijai/WanVideo_comfy_fp8_scaled",
     "Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
     "diffusion_models", "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"),
    ("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors",
     "vae", "Wan2_1_VAE_bf16.safetensors"),
    ("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors",
     "text_encoders", "umt5-xxl-enc-bf16.safetensors"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/clip_vision/clip_vision_h.safetensors",
     "clip_vision", "clip_vision_h.safetensors"),
    ("Kijai/WanVideo_comfy", "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors",
     "loras", "WanAnimate_relight_lora_fp16.safetensors"),
    ("Kijai/WanVideo_comfy",
     "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
     "loras", "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
    # SAM3 (non-gated fp16 mirror) for headless person segmentation in replace mode.
    ("yolain/sam3-safetensors", "sam3-fp16.safetensors",
     "sam3", "sam3-fp16.safetensors"),
]


@app.function(image=image, volumes={"/models": volume}, timeout=3600)
def _download_models() -> str:
    import os
    import shutil

    from huggingface_hub import hf_hub_download

    out = []
    for repo, path, subdir, name in MODELS:
        dest_dir = os.path.join(COMFY_MODELS, subdir)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
            out.append(f"skip (exists): {subdir}/{name}")
            continue
        src = hf_hub_download(repo_id=repo, filename=path)
        shutil.copyfile(src, dest)
        out.append(f"got {subdir}/{name}  ({os.path.getsize(dest)//(1024*1024)} MB)")
    volume.commit()
    return "\n".join(out)


@app.local_entrypoint()
def download_models() -> None:
    print(_download_models.remote())


def _start_comfy_server(port: int = 8188):
    """Launch the ComfyUI server (proper CWD/sys.path -> custom nodes load right)
    and block until /object_info responds. Returns the Popen handle."""
    import json
    import os
    import subprocess
    import time
    import urllib.request

    # Point ComfyUI at the models on the mounted volume.
    os.makedirs(COMFY_MODELS, exist_ok=True)
    with open(os.path.join(COMFY, "extra_model_paths.yaml"), "w") as f:
        f.write(
            "wan_volume:\n"
            f"  base_path: {COMFY_MODELS}/\n"
            "  diffusion_models: diffusion_models\n"
            "  vae: vae\n"
            "  text_encoders: text_encoders\n"
            "  clip_vision: clip_vision\n"
            "  loras: loras\n"
        )
    # sam3 is a custom model-folder type; symlink it in directly so LoadSam3Model
    # finds it regardless of folder_paths registration.
    os.makedirs(f"{COMFY_MODELS}/sam3", exist_ok=True)
    link = f"{COMFY}/models/sam3"
    if not os.path.islink(link) and not os.path.isdir(link):
        os.makedirs(f"{COMFY}/models", exist_ok=True)
        try:
            os.symlink(f"{COMFY_MODELS}/sam3", link)
        except FileExistsError:
            pass

    proc = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", str(port),
         "--disable-auto-launch"],
        cwd=COMFY,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI server exited early: {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base}/object_info", timeout=2) as r:
                if r.status == 200:
                    json.loads(r.read())
                    return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("ComfyUI server did not become ready in time")


@app.function(image=image, gpu="H100", volumes={"/models": volume}, timeout=900)
def _list_nodes(dump: str = "") -> str:
    """Start the server, verify node classes, and optionally dump input/output
    schemas for the comma-separated node names in `dump` (to hand-build a workflow)."""
    import json
    import urllib.request

    proc = _start_comfy_server()
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=60) as r:
            info = json.loads(r.read())
    finally:
        proc.terminate()
    need = [
        "WanVideoModelLoader", "WanVideoSampler", "WanVideoAnimateEmbeds",
        "WanVideoVAELoader", "WanVideoClipVisionEncode", "WanVideoTextEncodeCached",
        "DWPreprocessor", "Sam2Segmentation", "DownloadAndLoadSAM2Model",
        "VHS_LoadVideo", "VHS_VideoCombine", "ImageResizeKJv2", "PointsEditor",
        "WanVideoLoraSelectMulti",
    ]
    missing = [n for n in need if n not in info]
    out = [
        f"total nodes registered: {len(info)}",
        f"missing workflow nodes ({len(missing)}): {missing}",
    ]
    for name in [d for d in dump.split(",") if d]:
        n = info.get(name)
        if not n:
            out.append(f"\n## {name}: NOT FOUND")
            continue
        out.append(f"\n## {name}")
        out.append("  required: " + json.dumps(n["input"].get("required", {}), ensure_ascii=False))
        out.append("  optional: " + json.dumps(n["input"].get("optional", {}), ensure_ascii=False))
        out.append(f"  outputs: {n.get('output_name', n.get('output'))}")
    return "\n".join(out)


@app.local_entrypoint()
def list_nodes(dump: str = "") -> None:
    print(_list_nodes.remote(dump))


# Model filenames (flat, as downloaded to the volume's ComfyUI model dirs).
DIT = "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
T5 = "umt5-xxl-enc-bf16.safetensors"
CLIPV = "clip_vision_h.safetensors"
LORA_RELIGHT = "WanAnimate_relight_lora_fp16.safetensors"
LORA_LIGHTX2V = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
       "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
       "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
       "杂乱的背景，三条腿，背景人很多，倒着走")


def build_workflow(img_name, vid_name, prompt, width, height, frame_cap, seed):
    """Minimal pose-driven Wan2.2-Animate API graph (no SAM2/mask/face).
    fp8 DiT + relight + lightx2v 6-step distill LoRA. num_frames is wired from
    VHS frame_count (out 1) so it always matches the real pose frames (no reverse)."""
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
            "scheduler": "dpm++_sde", "riflex_freq_index": 0,
            "text_embeds": ["text", 0]}},
        "decode": {"class_type": "WanVideoDecode", "inputs": {
            "vae": ["vae", 0], "samples": ["sampler", 0], "enable_vae_tiling": False,
            "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}},
        # WanAnim looping rounds up to whole frame_window_size windows and reflect-pads
        # the pose tail (-> reversed tail). Trim back to the real loaded frame count.
        "trim": {"class_type": "GetImageRangeFromBatch", "inputs": {
            "images": ["decode", 0], "start_index": 0, "num_frames": ["load_vid", 1]}},
        "save": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["trim", 0], "frame_rate": 16, "loop_count": 0,
            "filename_prefix": "wananim", "format": "video/h264-mp4",
            "pingpong": False, "save_output": True}},
    }


@app.function(image=image, gpu="H100", volumes={"/models": volume}, timeout=1800)
def _run_animate(img_bytes: bytes, vid_bytes: bytes, prompt: str,
                 width: int = 832, height: int = 480, num_frames: int = 77,
                 seed: int = 42) -> dict:
    import json
    import os
    import time
    import urllib.request

    os.makedirs(f"{COMFY}/input", exist_ok=True)
    with open(f"{COMFY}/input/ref.png", "wb") as f:
        f.write(img_bytes)
    with open(f"{COMFY}/input/drive.mp4", "wb") as f:
        f.write(vid_bytes)

    proc = _start_comfy_server()
    base = "http://127.0.0.1:8188"
    try:
        wf = build_workflow("ref.png", "drive.mp4", prompt, width, height, num_frames, seed)
        body = json.dumps({"prompt": wf}).encode()
        req = urllib.request.Request(f"{base}/prompt", data=body,
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        pid = resp["prompt_id"]
        # Poll history until this prompt finishes.
        out = None
        for _ in range(1500):
            time.sleep(1)
            with urllib.request.urlopen(f"{base}/history/{pid}", timeout=10) as r:
                hist = json.loads(r.read())
            if pid in hist:
                h = hist[pid]
                status = h.get("status", {})
                if status.get("status_str") == "error" or status.get("completed") is False and h.get("status", {}).get("messages"):
                    pass
                outs = h.get("outputs", {})
                if outs:
                    out = outs
                    break
                if status.get("status_str") == "error":
                    return {"ok": False, "error": json.dumps(status)[:1500]}
        if not out:
            return {"ok": False, "error": "timed out waiting for workflow"}
        # Find the produced video file.
        for node_out in out.values():
            for key in ("gifs", "videos", "images"):
                for item in node_out.get(key, []):
                    fn = item.get("filename")
                    sub = item.get("subfolder", "")
                    typ = item.get("type", "output")
                    d = {"output": "output", "temp": "temp"}.get(typ, "output")
                    path = os.path.join(COMFY, d, sub, fn)
                    if fn and fn.endswith((".mp4", ".webm")) and os.path.isfile(path):
                        return {"ok": True, "video": open(path, "rb").read(),
                                "filename": fn}
        return {"ok": False, "error": "no video output: " + json.dumps(out)[:1500]}
    finally:
        proc.terminate()


@app.local_entrypoint()
def run_animate(image: str, video: str, prompt: str = "a person dancing",
                width: int = 832, height: int = 480, frames: int = 77) -> None:
    with open(image, "rb") as f:
        ib = f.read()
    with open(video, "rb") as f:
        vb = f.read()
    res = _run_animate.remote(ib, vb, prompt, width, height, frames)
    if res.get("ok"):
        with open("/tmp/wananim_out.mp4", "wb") as f:
            f.write(res["video"])
        print(f"OK -> /tmp/wananim_out.mp4 ({len(res['video'])} bytes), src={res['filename']}")
    else:
        print("FAIL:", res.get("error"))


# ---------------------------------------------------------------------------
# Replace mode: swap the person in the driving video with the character image,
# keeping the original scene. SAM3 (text="person") auto-segments + tracks the
# person -> mask; bg_images = original frames -> WanVideoAnimateEmbeds inpaints.
# ---------------------------------------------------------------------------
def build_replace_workflow(img_name, vid_name, prompt, width, height, num_frames, seed):
    wf = build_workflow(img_name, vid_name, prompt, width, height, num_frames, seed)
    # Force driving frames to the target resolution so SAM3 mask / DWPose / bg all align.
    wf["load_vid"]["inputs"]["custom_width"] = width
    wf["load_vid"]["inputs"]["custom_height"] = height
    # SAM3 person segmentation + tracking across frames.
    wf["sam3_model"] = {"class_type": "easy sam3ModelLoader", "inputs": {
        "model": "sam3-fp16.safetensors", "segmentor": "video",
        "device": "cuda", "precision": "fp16"}}
    wf["sam3_seg"] = {"class_type": "easy sam3VideoSegmentation", "inputs": {
        "sam3_model": ["sam3_model", 0], "video_frames": ["load_vid", 0],
        "prompt": "person", "frame_index": 0, "object_id": 1,
        "score_threshold_detection": 0.5, "new_det_thresh": 0.7,
        "propagation_direction": "both", "start_frame_index": 0,
        "max_frames_to_track": -1, "close_after_propagation": True,
        "keep_model_loaded": False}}
    # Official mask pipeline: SAM -> GrowMask -> BlockifyMask. Blockify turns the tight
    # silhouette into coarse blocks so the (differently-shaped) character has room and
    # doesn't get clipped to the original person's outline -> no black edges.
    wf["grow"] = {"class_type": "GrowMask", "inputs": {
        "mask": ["sam3_seg", 0], "expand": 10, "tapered_corners": True}}
    wf["blockify"] = {"class_type": "BlockifyMask", "inputs": {
        "masks": ["grow", 0], "block_size": 32, "device": "gpu"}}
    # bg_images must have the masked region BLACKED OUT, otherwise the original person
    # leaks through bg conditioning and the reference character never takes.
    wf["bg_masked"] = {"class_type": "DrawMaskOnImage", "inputs": {
        "image": ["load_vid", 0], "mask": ["blockify", 0], "color": "0, 0, 0",
        "device": "gpu"}}
    wf["embeds"]["inputs"]["bg_images"] = ["bg_masked", 0]
    wf["embeds"]["inputs"]["mask"] = ["blockify", 0]
    wf["save"]["inputs"]["filename_prefix"] = "wanrepl"
    return wf


def _submit_and_fetch(base, wf):
    import json
    import os
    import time
    import urllib.error
    import urllib.request

    body = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(f"{base}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": "rejected: " + e.read().decode()[:1500]}
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
                return {"ok": False, "error": json.dumps(h.get("status"))[:1500]}
    if not out:
        return {"ok": False, "error": "timed out"}
    for node_out in out.values():
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key, []):
                fn, sub = item.get("filename"), item.get("subfolder", "")
                typ = item.get("type", "output")
                d = {"output": "output", "temp": "temp"}.get(typ, "output")
                path = os.path.join(COMFY, d, sub, fn or "")
                if fn and fn.endswith((".mp4", ".webm")) and os.path.isfile(path):
                    return {"ok": True, "video": open(path, "rb").read(), "filename": fn}
    return {"ok": False, "error": "no video: " + json.dumps(out)[:1500]}


@app.function(image=image, gpu="H100", volumes={"/models": volume}, timeout=2400)
def _run_replace(img_bytes: bytes, vid_bytes: bytes, prompt: str,
                 seed: int = 42) -> dict:
    import os

    import cv2

    os.makedirs(f"{COMFY}/input", exist_ok=True)
    with open(f"{COMFY}/input/ref.png", "wb") as f:
        f.write(img_bytes)
    drive = f"{COMFY}/input/drive.mp4"
    with open(drive, "wb") as f:
        f.write(vid_bytes)

    # Replace output resolution = driving video's, aligned to 16; frames from the video.
    cap = cv2.VideoCapture(drive)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 832)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    W = max(16, int(round(vw / 16.0)) * 16)
    H = max(16, int(round(vh / 16.0)) * 16)
    avail = int(n / max(fps, 1.0) * 16) if n else 49
    frames = ((max(13, min(81, avail)) - 1) // 4) * 4 + 1

    proc = _start_comfy_server()
    try:
        wf = build_replace_workflow("ref.png", "drive.mp4", prompt, W, H, frames, seed)
        res = _submit_and_fetch("http://127.0.0.1:8188", wf)
        res["meta"] = {"width": W, "height": H, "frames": frames}
        return res
    finally:
        proc.terminate()


@app.function(image=image, gpu="H100", volumes={"/models": volume}, timeout=1200)
def _debug_sam3(vid_bytes: bytes, text: str = "person", width: int = 832,
                height: int = 480) -> dict:
    """Run ONLY SAM3 on the driving video and return the mask as a video, to verify
    the person is actually being segmented."""
    import os

    os.makedirs(f"{COMFY}/input", exist_ok=True)
    with open(f"{COMFY}/input/drive.mp4", "wb") as f:
        f.write(vid_bytes)
    wf = {
        "load_vid": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": "drive.mp4", "force_rate": 16, "custom_width": width,
            "custom_height": height, "frame_load_cap": 81, "skip_first_frames": 0,
            "select_every_nth": 1}},
        "sam3_model": {"class_type": "easy sam3ModelLoader", "inputs": {
            "model": "sam3-fp16.safetensors", "segmentor": "video",
            "device": "cuda", "precision": "fp16"}},
        "sam3_seg": {"class_type": "easy sam3VideoSegmentation", "inputs": {
            "sam3_model": ["sam3_model", 0], "video_frames": ["load_vid", 0],
            "prompt": text, "frame_index": 0, "object_id": 1,
            "score_threshold_detection": 0.5, "new_det_thresh": 0.7,
            "propagation_direction": "both", "start_frame_index": 0,
            "max_frames_to_track": -1, "close_after_propagation": True,
            "keep_model_loaded": False}},
        "mask_img": {"class_type": "MaskToImage", "inputs": {"mask": ["sam3_seg", 0]}},
        "save": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["mask_img", 0], "frame_rate": 16, "loop_count": 0,
            "filename_prefix": "maskdbg", "format": "video/h264-mp4",
            "pingpong": False, "save_output": True}},
    }
    proc = _start_comfy_server()
    try:
        return _submit_and_fetch("http://127.0.0.1:8188", wf)
    finally:
        proc.terminate()


@app.function(image=image, gpu="H100", volumes={"/models": volume}, timeout=1800)
def _debug_frames(img_bytes: bytes, vid_bytes: bytes, prompt: str) -> dict:
    """Run the replace graph but ALSO save first-frame PNGs of the SAM3 mask and the
    output, so we can eyeball mask polarity and what actually got generated."""
    import json
    import os
    import time
    import urllib.request

    os.makedirs(f"{COMFY}/input", exist_ok=True)
    with open(f"{COMFY}/input/ref.png", "wb") as f:
        f.write(img_bytes)
    drive = f"{COMFY}/input/drive.mp4"
    with open(drive, "wb") as f:
        f.write(vid_bytes)

    import cv2
    cap = cv2.VideoCapture(drive)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 832)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    cap.release()
    W = max(16, int(round(vw / 16.0)) * 16)
    H = max(16, int(round(vh / 16.0)) * 16)

    wf = build_replace_workflow("ref.png", "drive.mp4", prompt, W, H, 200, 42)
    wf["mask_img"] = {"class_type": "MaskToImage", "inputs": {"mask": ["grow", 0]}}
    wf["save_mask"] = {"class_type": "SaveImage", "inputs": {
        "images": ["mask_img", 0], "filename_prefix": "dbgmask"}}
    wf["save_out"] = {"class_type": "SaveImage", "inputs": {
        "images": ["trim", 0], "filename_prefix": "dbgout"}}

    proc = _start_comfy_server()
    base = "http://127.0.0.1:8188"
    try:
        body = json.dumps({"prompt": wf}).encode()
        req = urllib.request.Request(f"{base}/prompt", data=body,
                                     headers={"Content-Type": "application/json"})
        pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
        out = None
        for _ in range(1800):
            time.sleep(1)
            with urllib.request.urlopen(f"{base}/history/{pid}", timeout=10) as r:
                hist = json.loads(r.read())
            if pid in hist and hist[pid].get("outputs"):
                out = hist[pid]["outputs"]
                break
            if pid in hist and hist[pid].get("status", {}).get("status_str") == "error":
                return {"error": json.dumps(hist[pid]["status"])[:1500]}
        outs = []
        res = {}
        for node_out in (out or {}).values():
            for item in node_out.get("images", []):
                fn, sub = item.get("filename", ""), item.get("subfolder", "")
                path = os.path.join(COMFY, "output", sub, fn)
                if not os.path.isfile(path):
                    continue
                if fn.startswith("dbgmask") and "mask" not in res:
                    res["mask"] = open(path, "rb").read()
                elif fn.startswith("dbgout"):
                    outs.append((fn, path))
        outs.sort()
        if outs:  # first / middle / last output frame, to spot a reversed tail
            for tag, (_, p) in (("out_first", outs[0]),
                                ("out_mid", outs[len(outs) // 2]),
                                ("out_last", outs[-1])):
                res[tag] = open(p, "rb").read()
            res["n_out"] = len(outs)
        return res
    finally:
        proc.terminate()


@app.local_entrypoint()
def debug_frames(image: str, video: str, prompt: str = "a person") -> None:
    with open(image, "rb") as f:
        ib = f.read()
    with open(video, "rb") as f:
        vb = f.read()
    res = _debug_frames.remote(ib, vb, prompt)
    if res.get("error"):
        print("ERR:", res["error"])
        return
    print("n_out frames:", res.get("n_out"))
    for k in ("mask", "out_first", "out_mid", "out_last"):
        if res.get(k):
            p = f"/tmp/dbg_{k}.png"
            with open(p, "wb") as f:
                f.write(res[k])
            print(f"{k} -> {p} ({len(res[k])} bytes)")


@app.local_entrypoint()
def debug_sam3(video: str, text: str = "person") -> None:
    with open(video, "rb") as f:
        vb = f.read()
    res = _debug_sam3.remote(vb, text)
    if res.get("ok"):
        with open("/tmp/sam3_mask.mp4", "wb") as f:
            f.write(res["video"])
        print(f"OK -> /tmp/sam3_mask.mp4 ({len(res['video'])} bytes)")
    else:
        print("FAIL:", res.get("error"))


@app.local_entrypoint()
def run_replace(image: str, video: str, prompt: str = "a person",
                seed: int = 42) -> None:
    with open(image, "rb") as f:
        ib = f.read()
    with open(video, "rb") as f:
        vb = f.read()
    res = _run_replace.remote(ib, vb, prompt, seed)
    if res.get("ok"):
        with open("/tmp/wanrepl_out.mp4", "wb") as f:
            f.write(res["video"])
        print(f"OK -> /tmp/wanrepl_out.mp4 ({len(res['video'])} bytes), "
              f"src={res['filename']}, meta={res.get('meta')}")
    else:
        print("FAIL:", res.get("error"))
