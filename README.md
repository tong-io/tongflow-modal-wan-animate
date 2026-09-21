# tongflow-modal-wan-animate

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. End-to-end character animation with **Wan-Animate-2** (`Wan-AI/Wan2.2-Animate-2-14B`), running on a single GPU via [Modal](https://modal.com). Takes a character image plus a driving video and reenacts the driving motion onto the character.

## Capabilities

- **Motion transfer** (`video-image-gen-video-move`) — retarget a driving video's motion onto a character image.

Wan-Animate-2 consumes the driving video directly, so there is no pose/face extraction stage: identity and motion come out of one redesigned DiT. It also has **no character-replacement mode** — unlike this plugin's Wan-Animate v1 releases, it no longer serves `video-image-gen-video-mix`; [tongflow-modal-scail2](https://github.com/tong-io/tongflow-modal-scail2) covers that slot.

## Engine

Runs the **distillation** checkpoint (`wan_animate_2_bf16_distillation.safetensors`) through [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)'s `WanVideoPipeline`: 10 steps, no CFG, one A100-80GB. Upstream's own repo only ships an 8-GPU FSDP pipeline and the diffusers integration is still an unmerged PR, so DiffSynth is the one maintained single-GPU path.

Output resolution follows the character image (aligned to 16, capped at the 720×1280 pixel budget); the driving video is letterboxed to match, resampled to the output FPS, and generated clip by clip with a one-frame overlap so identity and motion carry across the seams.

### Advanced

Under the node's collapsed **Advanced** section:

| Knob | Default | Notes |
| --- | --- | --- |
| Steps | `10` | What the distillation checkpoint is trained for. |
| CFG scale | `1.0` | The distilled checkpoint expects 1.0; raising it also needs more steps. |
| Shift | `5.0` | Flow-matching sigma shift. |
| Output FPS | `24` | Also the rate the driving video is resampled to. |

## Prompt

Wan-Animate-2 expects a caption of the character's **appearance and background**, not of the motion (the motion comes from the driving video). Upstream's recommended shape:

```
人物外观描述：<what the character wears / looks like>
背景描述：<the scene>
```

Leave the node's text empty and a neutral "keep the reference image's appearance and scene" caption is used.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |
| `HF_TOKEN` | — | Optional. Both weight repos are public; a token only helps with Hugging Face rate limits. |

### Weights (Hugging Face)

`modal run download.py::download` pulls, into the shared `models` volume:

- `Wan-AI/Wan2.2-Animate-2-14B` — the distillation DiT, UMT5 text encoder, CLIP vision encoder and UMT5 tokenizer (the base checkpoint and the FLUX retarget model are skipped);
- `Wan-AI/Wan2.1-T2V-14B` — `Wan2.1_VAE.pth`, matching DiffSynth-Studio's own Animate-2 example.

Both are Apache 2.0 and ungated.
