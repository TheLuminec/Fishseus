"""
vision_service.py

Thin camera capture + image-understanding service for Fishseus.

Responsibilities:
- Capture still frames from named camera sources:
    * "command" sources run a local capture binary (rpicam-still, fswebcam, ...)
    * "url" sources fetch a JPEG snapshot over HTTP (network cams, the future
      nerf-turret stream, anything that can serve a frame)
- Describe frames with a vision-capable LLM through an OpenAI-compatible
  endpoint (Ollama vision models such as qwen2.5vl, llava, moondream)

Non-responsibilities:
- No motion logic, no assistant logic, no GPIO

Config shape (config/fish_config.json "vision" section):
    {
      "endpoint_url": "http://ollama.angelfish-gamma.ts.net/v1/chat/completions",
      "model": "qwen2.5vl:3b",
      "timeout_s": 60,
      "capture_dir": "../tmp/vision",
      "cameras": {
        "front":  {"type": "command", "command": "rpicam-still -o {output} --width 1280 --height 720 --nopreview -t 500"},
        "turret": {"type": "url", "url": "http://turret.local:8080/snapshot.jpg"}
      }
    }

Example:
    vision = VisionService(VisionConfig(cameras={...}))
    vision.initialize()
    print(vision.look("front", "What do you see?"))
    vision.shutdown()

Orchestrator usage:
    vision = VisionService(VisionConfig(**config.get("vision", {})))
    vision.initialize()
    ...
    vision.shutdown()
"""

from __future__ import annotations

import base64
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

from services import Service, ServiceConfig, ServiceError, ROOT_DIR


class VisionServiceError(ServiceError):
    pass


@dataclass(frozen=True)
class VisionConfig(ServiceConfig):
    module_name: str = "vision"

    # OpenAI-compatible chat endpoint that accepts image_url content parts.
    endpoint_url: str = "http://ollama.angelfish-gamma.ts.net/v1/chat/completions"
    model: str = "qwen2.5vl:3b"
    api_key: Optional[str] = None
    timeout_s: float = 60.0

    max_tokens: int = 300
    temperature: float = 0.4

    # Where captured frames are written.
    capture_dir: Path = ROOT_DIR / "tmp" / "vision"

    # Named camera sources.  type: "command" (local capture binary) or
    # "url" (HTTP JPEG snapshot).  {output} in a command is replaced with
    # the destination file path.
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Camera used when the caller doesn't name one.
    default_camera: str = ""

    def validate(self) -> bool:
        if not self.endpoint_url:
            raise VisionServiceError("endpoint_url must be set")
        if self.timeout_s <= 0:
            raise VisionServiceError(f"timeout_s must be positive: {self.timeout_s}")
        return True


class VisionService(Service):
    def __init__(self, config: VisionConfig = VisionConfig()) -> None:
        self.config = config
        self.config.capture_dir.mkdir(parents=True, exist_ok=True)
        self._session: Optional[requests.Session] = requests.Session()
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.config.validate()
        self.config.capture_dir.mkdir(parents=True, exist_ok=True)
        if self._session is None:
            self._session = requests.Session()
        self._initialized = True

    def shutdown(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._initialized = False

    def reset(self) -> bool:
        self.shutdown()
        self.initialize()
        return True

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "service": "ok" if self._initialized else "uninitialized",
            "cameras": self.available_cameras(),
            "default_camera": self.config.default_camera,
        }

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def available_cameras(self) -> list[str]:
        return sorted(self.config.cameras.keys())

    def capture(self, camera: str = "") -> Path:
        """
        Capture one frame from the named camera and return the JPEG path.
        """
        name = camera or self.config.default_camera
        if not name:
            names = self.available_cameras()
            if not names:
                raise VisionServiceError("No cameras configured")
            name = names[0]

        cam = self.config.cameras.get(name)
        if cam is None:
            raise VisionServiceError(f"Unknown camera '{name}' (have: {self.available_cameras()})")

        output = self.config.capture_dir / f"frame_{name}_{int(time.time() * 1000)}.jpg"

        cam_type = cam.get("type", "command")
        if cam_type == "command":
            self._capture_command(cam, output)
        elif cam_type == "url":
            self._capture_url(cam, output)
        else:
            raise VisionServiceError(f"Unknown camera type '{cam_type}' for '{name}'")

        if not output.exists() or output.stat().st_size == 0:
            raise VisionServiceError(f"Capture from '{name}' produced no image")

        return output

    def _capture_command(self, cam: dict[str, Any], output: Path) -> None:
        template = cam.get("command", "")
        if not template:
            raise VisionServiceError("Camera has no capture command configured")

        cmd = [
            part.replace("{output}", str(output))
            for part in shlex.split(template)
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=float(cam.get("timeout_s", 15.0)),
            )
        except subprocess.TimeoutExpired as exc:
            raise VisionServiceError("Camera capture timed out") from exc
        except FileNotFoundError as exc:
            raise VisionServiceError(f"Capture binary not found: {cmd[0]}") from exc

        if proc.returncode != 0:
            raise VisionServiceError(
                f"Capture command failed ({proc.returncode}): {proc.stderr[:300]}"
            )

    def _capture_url(self, cam: dict[str, Any], output: Path) -> None:
        url = cam.get("url", "")
        if not url:
            raise VisionServiceError("Camera has no snapshot URL configured")

        session = self._session or requests
        try:
            resp = session.get(url, timeout=float(cam.get("timeout_s", 10.0)))
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise VisionServiceError(f"Snapshot fetch failed: {exc}") from exc

        output.write_bytes(resp.content)

    # ------------------------------------------------------------------
    # Understanding
    # ------------------------------------------------------------------

    def describe(self, image_path: str | Path, prompt: str = "Describe what you see in one or two sentences.") -> str:
        """
        Send the image to the vision model and return its text description.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise VisionServiceError(f"Image not found: {image_path}")

        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
        }

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        session = self._session or requests
        try:
            resp = session.post(
                self.config.endpoint_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout_s,
            )
        except requests.RequestException as exc:
            raise VisionServiceError(f"Vision model request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise VisionServiceError(
                f"Vision model returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise VisionServiceError(f"Unexpected vision response shape: {resp.text[:300]}") from exc

        return str(content).strip()

    def look(self, camera: str = "", prompt: str = "Describe what you see in one or two sentences.") -> str:
        """Capture + describe in one call. Convenience for tools."""
        frame = self.capture(camera)
        return self.describe(frame, prompt)
