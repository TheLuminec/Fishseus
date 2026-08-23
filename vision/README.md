# Vision Module

Thin camera-capture + image-understanding service for Fishseus. Grabs a still
frame from a named camera and describes it with a vision-capable LLM over an
OpenAI-compatible endpoint.

**Responsibilities:** capture frames (local capture binary or HTTP snapshot),
describe frames with a vision model.

**Non-responsibilities:** no motion, assistant, or GPIO logic.

## Configuration (`VisionConfig`)

| Field            | Default                        | Purpose                                          |
| ---------------- | ------------------------------ | ------------------------------------------------ |
| `module_name`    | `"vision"`                     | Service key in the orchestrator config.          |
| `endpoint_url`   | Ollama `…/v1/chat/completions` | Vision-capable chat endpoint (accepts images).   |
| `model`          | `"qwen2.5vl:3b"`               | Vision model name.                               |
| `api_key`        | `None`                         | Optional bearer token.                           |
| `timeout_s`      | `60.0`                         | Request timeout.                                 |
| `max_tokens`     | `300`                          | Description length cap.                          |
| `temperature`    | `0.4`                          | Sampling temperature.                            |
| `capture_dir`    | `<root>/tmp/vision`            | Where captured frames are written.               |
| `cameras`        | `{}`                           | Named sources (see below).                       |
| `default_camera` | `""`                           | Camera used when none is named.                  |

Each camera is `{"type": "command", "command": "... {output} ..."}` (runs a local
capture binary; `{output}` is replaced with the destination path) or
`{"type": "url", "url": "http://.../snapshot.jpg"}` (fetches a JPEG over HTTP).

`config.validate()` raises `VisionServiceError` if `endpoint_url` is empty or
`timeout_s <= 0`. It runs in `initialize()`.

## Lifecycle API

- `initialize()` – validate config, ensure `capture_dir`, open the HTTP session.
- `shutdown()` – close the session.
- `reset()` – `shutdown()` then `initialize()`.
- `status()` – `{enabled, service, cameras, default_camera}`.

## Vision API

- `available_cameras()` – configured camera names.
- `capture(camera="") -> Path` – capture one frame; returns the JPEG path.
- `describe(image_path, prompt=…) -> str` – describe an image with the model.
- `look(camera="", prompt=…) -> str` – capture + describe in one call (for tools).

## How it works

- `capture` runs the camera's command (with `{output}` substituted) or fetches
  its snapshot URL, then verifies a non-empty JPEG was produced.
- `describe` base64-encodes the frame into an `image_url` content part and posts
  it to the endpoint, returning `choices[0].message.content`.

## Usage

```python
vision = VisionService(VisionConfig(
    cameras={"front": {"type": "command",
                       "command": "rpicam-still -o {output} --nopreview -t 500"}},
    default_camera="front",
))
vision.initialize()
print(vision.look("front", "What do you see?"))
vision.shutdown()
```

## Requirements

- A reachable vision-capable OpenAI-compatible endpoint, `requests`, and a
  working capture binary (e.g. `rpicam-still`, `fswebcam`) or snapshot URL.
