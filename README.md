# tongflow-modal-wan-animate

Official TongFlow plugin. Character swap and motion transfer with **Wan-Animate** (`Wan-AI/Wan2.2-Animate-14B`), running on a GPU via [Modal](https://modal.com). Takes a driving video plus a reference and produces Animate Mix / Animate Move-style output.

## Capabilities

- **Character swap** (`video-image-gen-video-mix`) — replace the character / blend the scene (Animate Mix).
- **Motion transfer** (`video-image-gen-video-move`) — retarget motion onto a reference (Animate Move).

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |
| `HF_TOKEN` | ✅ | Required to fetch `Wan-AI/Wan2.2-Animate-14B` from Hugging Face. |

### Weights (Hugging Face)

The plugin injects `HF_TOKEN` from your TongFlow Settings into the Modal download job at deploy time — no manual `modal secret create` needed. Without it the weight download fails.
