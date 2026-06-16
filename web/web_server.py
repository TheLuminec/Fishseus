#!/usr/bin/env python3
"""
web_server.py

Fishseus web control panel backed by the real project services.

This server intentionally avoids Flask so it can run with only the Python
standard library. It serves `fishseus_ui.html`, reads/writes
`fish_config.json`, and exposes JSON endpoints that use the real Fishseus
modules when available:

- llm/llm_service.py
- assistant/assistant_service.py
- motion/motion_service.py
- tts/tts_service.py

Run from the project root or from a `web/` folder:

    python3 web_server.py 8000

Then open:

    http://<pi-ip>:8000/

Notes:
- Motion and TTS services are initialized lazily the first time they are used.
- If you change motion or TTS config from the UI, the corresponding runtime
  service is torn down and rebuilt on next use.
- This is intended for trusted LAN/Tailscale use only. Do not expose publicly
  without authentication.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
import traceback
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------
# Paths and import setup
# ---------------------------------------------------------------------

SERVER_DIR = Path(__file__).resolve().parent

# Support both:
#   fishseus/web/web_server.py
#   fishseus/web_server.py
if (SERVER_DIR / "assistant").exists() and (SERVER_DIR / "motion").exists():
    PROJECT_ROOT = SERVER_DIR
else:
    PROJECT_ROOT = SERVER_DIR.parent

UI_FILE_CANDIDATES = [
    SERVER_DIR / "fishseus_ui.html",
    SERVER_DIR / "templates" / "fishseus_ui.html",
    PROJECT_ROOT / "web" / "fishseus_ui.html",
    PROJECT_ROOT / "fishseus_ui.html",
]

CONFIG_FILE = PROJECT_ROOT / "config" / "fish_config.json"
PERSONALITY_FILE = PROJECT_ROOT / "config" / "personality_prompt.txt"

for subdir in ["audio", "stt", "llm", "assistant", "motion", "tts"]:
    candidate = PROJECT_ROOT / subdir
    if candidate.exists():
        sys.path.insert(0, str(candidate))


# Real service imports. These intentionally do not fall back to stubs.
# Import failures are captured and returned through /api/status.
SERVICE_IMPORT_ERRORS: dict[str, str] = {}

try:
    from llm_service import LlmConfig, LlmService
except Exception as exc:  # pragma: no cover - runtime environment dependent
    LlmConfig = None  # type: ignore[assignment]
    LlmService = None  # type: ignore[assignment]
    SERVICE_IMPORT_ERRORS["llm"] = f"{type(exc).__name__}: {exc}"

try:
    from assistant_service import AssistantConfig, AssistantService, Tool, ToolRegistry
except Exception as exc:  # pragma: no cover - runtime environment dependent
    AssistantConfig = None  # type: ignore[assignment]
    AssistantService = None  # type: ignore[assignment]
    Tool = None  # type: ignore[assignment]
    ToolRegistry = None  # type: ignore[assignment]
    SERVICE_IMPORT_ERRORS["assistant"] = f"{type(exc).__name__}: {exc}"

try:
    from motion_service import MotorConfig, MotionService
except Exception as exc:  # pragma: no cover - likely unavailable off Pi
    MotorConfig = None  # type: ignore[assignment]
    MotionService = None  # type: ignore[assignment]
    SERVICE_IMPORT_ERRORS["motion"] = f"{type(exc).__name__}: {exc}"

try:
    from tts_service import TtsConfig, TtsService
except Exception as exc:  # pragma: no cover - runtime environment dependent
    TtsConfig = None  # type: ignore[assignment]
    TtsService = None  # type: ignore[assignment]
    SERVICE_IMPORT_ERRORS["tts"] = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_PERSONALITY = """You are Fishseus, a dramatic animatronic fish oracle.

You are witty, strange, sarcastic, and helpful.
You speak like a washed-up sea prophet trapped in a novelty wall fish.
Keep responses short because they are spoken aloud.
Prefer 1-3 sentences.

You must respond ONLY as valid JSON with this exact shape:
{
  "speak": "short text to say aloud",
  "motion": "idle | speaking | happy | annoyed | thinking | excited",
  "tool_calls": [],
  "memory_updates": []
}

Rules:
- The "speak" field must always be present and non-empty.
- Use "tool_calls" only when useful.
- Use "memory_updates" only when the user explicitly asks you to remember something.
- Keep JSON valid. No markdown. No code fences.
"""

DEFAULT_CONFIG: dict[str, Any] = {
    "audio": {
        "device": "plughw:3,0",
        "sample_rate": 16000,
        "channels": 1,
        "speech_threshold": 0.003,
        "silence_threshold": 0.0008,
        "silence_timeout_s": 1.0,
        "max_record_seconds": 12.0,
        "pre_roll_seconds": 0.4,
    },
    "stt": {
        "whisper_binary": str(PROJECT_ROOT / "stt" / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
        "model_path": str(PROJECT_ROOT / "stt" / "whisper.cpp" / "models" / "ggml-base.en.bin"),
        "threads": 4,
        "wake_words": ["fish", "fishseus", "hey fish"],
    },
    "llm": {
        "endpoint_url": "http://ollama.angelfish-gamma.ts.net/v1/chat/completions",
        "model": "qwen2.5:3b",
        "timeout_s": 45.0,
        "retries": 1,
        "temperature": 0.7,
        "max_tokens": 512,
        "disable_reasoning": True,
    },
    "tts": {
        "piper_binary": str(PROJECT_ROOT / "tts" / ".venv" / "bin" / "piper"),
        "voices_dir": str(PROJECT_ROOT / "tts" / "voices"),
        "default_voice": "en_US-arctic-medium",
        "audio_device": "plughw:0,0",
        "output_dir": str(PROJECT_ROOT / "tmp" / "tts"),
        "timeout_s": 30.0,
    },
    "assistant": {
        "assistant_name": "Fishseus",
        "user_name": "Caleb",
        "personality_path": str(PERSONALITY_FILE),
        "memory_path": str(PROJECT_ROOT / "data" / "assistant_memory.json"),
        "history_path": str(PROJECT_ROOT / "data" / "conversation_log.jsonl"),
        "max_history_turns": 8,
        "max_tool_calls": 3,
        "temperature": 0.7,
        "max_tokens": 512,
        "require_explicit_memory_intent": True,
    },
    "motion": {
        "pwm_frequency": 1000,
        "body_wiggle_time": 0.18,
        "tail_wiggle_time": 0.14,
        "mouth_open_time": 0.09,
        "mouth_close_time": 0.04,
        "envelope_window_s": 0.18,
        "motors": {
            "mouth": {
                "in1": 17,
                "in2": 27,
                "en": 22,
                "forward_speed": 82,
                "reverse_speed": 55,
                "neutral_return_time": 0.04,
            },
            "tail": {
                "in1": 23,
                "in2": 24,
                "en": 25,
                "forward_speed": 72,
                "reverse_speed": 48,
                "neutral_return_time": 0.03,
            },
            "body": {
                "in1": 5,
                "in2": 6,
                "en": 12,
                "forward_speed": 68,
                "reverse_speed": 45,
                "neutral_return_time": 0.03,
            },
        },
    },
    "web": {
        "enable_talk_tts_by_default": False,
        "enable_talk_motion_by_default": True,
    },
}


# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------

def deep_merge(default: Any, override: Any) -> Any:
    if isinstance(default, dict) and isinstance(override, dict):
        merged = dict(default)
        for key, value in override.items():
            merged[key] = deep_merge(default.get(key), value)
        return merged
    return override if override is not None else default


def load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return json.loads(json.dumps(DEFAULT_CONFIG))

    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return deep_merge(DEFAULT_CONFIG, data)
    except Exception:
        return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(config: dict[str, Any]) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


def ensure_files() -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "tmp").mkdir(parents=True, exist_ok=True)

    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)

    if not PERSONALITY_FILE.exists():
        PERSONALITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        PERSONALITY_FILE.write_text(DEFAULT_PERSONALITY, encoding="utf-8")


def get_ui_file() -> Optional[Path]:
    for candidate in UI_FILE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except Exception:
        return default


def clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return default


# ---------------------------------------------------------------------
# Runtime service manager
# ---------------------------------------------------------------------

class ServiceManager:
    """
    Owns the runtime instances used by the web control panel.

    The services are initialized lazily because:
    - importing/initializing GPIO should only happen when needed;
    - config changes should rebuild services cleanly;
    - the web UI should still serve even if a hardware service fails.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.llm: Any = None
        self.assistant: Any = None
        self.motion: Any = None
        self.tts: Any = None
        self.last_error: Optional[str] = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            cfg = load_config()
            voices: list[str] = []
            try:
                if TtsService is not None:
                    voices_dir = Path(cfg["tts"]["voices_dir"])
                    voices = sorted(p.stem for p in voices_dir.glob("*.onnx")) if voices_dir.exists() else []
            except Exception:
                voices = []

            return {
                "project_root": str(PROJECT_ROOT),
                "config_file": str(CONFIG_FILE),
                "personality_file": str(PERSONALITY_FILE),
                "ui_file": str(get_ui_file() or ""),
                "imports": {
                    "llm": "ok" if "llm" not in SERVICE_IMPORT_ERRORS else SERVICE_IMPORT_ERRORS["llm"],
                    "assistant": "ok" if "assistant" not in SERVICE_IMPORT_ERRORS else SERVICE_IMPORT_ERRORS["assistant"],
                    "motion": "ok" if "motion" not in SERVICE_IMPORT_ERRORS else SERVICE_IMPORT_ERRORS["motion"],
                    "tts": "ok" if "tts" not in SERVICE_IMPORT_ERRORS else SERVICE_IMPORT_ERRORS["tts"],
                },
                "runtime": {
                    "llm_initialized": self.llm is not None,
                    "assistant_initialized": self.assistant is not None,
                    "motion_initialized": self.motion is not None,
                    "tts_initialized": self.tts is not None,
                    "last_error": self.last_error,
                },
                "voices": voices,
                "model": cfg["llm"]["model"],
                "voice": cfg["tts"]["default_voice"],
                "audio_device": cfg["audio"]["device"],
                "tts_device": cfg["tts"]["audio_device"],
            }

    def reset_assistant(self) -> None:
        with self.lock:
            if self.llm is not None:
                try:
                    self.llm.close()
                except Exception:
                    pass
            self.llm = None
            self.assistant = None

    def reset_motion(self) -> None:
        with self.lock:
            if self.motion is not None:
                try:
                    self.motion.stop_all()
                    self.motion.shutdown()
                except Exception:
                    pass
            self.motion = None

    def reset_tts(self) -> None:
        with self.lock:
            self.tts = None

    def shutdown_all(self) -> None:
        self.reset_assistant()
        self.reset_motion()
        self.reset_tts()

    def get_llm(self) -> Any:
        if LlmConfig is None or LlmService is None:
            raise RuntimeError(f"llm_service import failed: {SERVICE_IMPORT_ERRORS.get('llm', 'unknown error')}")

        with self.lock:
            if self.llm is not None:
                return self.llm

            cfg = load_config()["llm"]
            self.llm = LlmService(
                LlmConfig(
                    endpoint_url=cfg["endpoint_url"],
                    model=cfg["model"],
                    api_key=cfg.get("api_key"),
                    timeout_s=float(cfg.get("timeout_s", 45.0)),
                    retries=int(cfg.get("retries", 1)),
                    retry_delay_s=float(cfg.get("retry_delay_s", 0.5)),
                    temperature=float(cfg.get("temperature", 0.7)),
                    max_tokens=int(cfg.get("max_tokens", 512)),
                    top_p=cfg.get("top_p"),
                    disable_reasoning=bool(cfg.get("disable_reasoning", True)),
                    extra_payload=cfg.get("extra_payload", {}),
                )
            )
            return self.llm

    def get_tts(self) -> Any:
        if TtsConfig is None or TtsService is None:
            raise RuntimeError(f"tts_service import failed: {SERVICE_IMPORT_ERRORS.get('tts', 'unknown error')}")

        with self.lock:
            if self.tts is not None:
                return self.tts

            cfg = load_config()["tts"]
            self.tts = TtsService(
                TtsConfig(
                    piper_binary=str(cfg["piper_binary"]),
                    voices_dir=Path(cfg["voices_dir"]),
                    default_voice=str(cfg["default_voice"]),
                    audio_device=str(cfg["audio_device"]),
                    output_dir=Path(cfg["output_dir"]),
                    timeout_s=float(cfg.get("timeout_s", 30.0)),
                )
            )
            return self.tts

    def get_motion(self) -> Any:
        if MotorConfig is None or MotionService is None:
            raise RuntimeError(f"motion_service import failed: {SERVICE_IMPORT_ERRORS.get('motion', 'unknown error')}")

        with self.lock:
            if self.motion is not None:
                return self.motion

            cfg = load_config()["motion"]
            motor_cfgs = {}
            for name, motor in cfg["motors"].items():
                motor_cfgs[name] = MotorConfig(
                    in1=int(motor["in1"]),
                    in2=int(motor["in2"]),
                    en=int(motor["en"]),
                    forward_speed=float(motor["forward_speed"]),
                    reverse_speed=float(motor["reverse_speed"]),
                    neutral_return_time=float(motor["neutral_return_time"]),
                )

            self.motion = MotionService(
                motors=motor_cfgs,
                pwm_frequency=int(cfg.get("pwm_frequency", 1000)),
                body_wiggle_time=float(cfg.get("body_wiggle_time", 0.18)),
                tail_wiggle_time=float(cfg.get("tail_wiggle_time", 0.14)),
                mouth_open_time=float(cfg.get("mouth_open_time", 0.09)),
                mouth_close_time=float(cfg.get("mouth_close_time", 0.04)),
                envelope_window_s=float(cfg.get("envelope_window_s", 0.18)),
            )
            self.motion.initialize()
            return self.motion

    def build_tool_registry(self) -> Any:
        if ToolRegistry is None or Tool is None:
            raise RuntimeError(f"assistant_service import failed: {SERVICE_IMPORT_ERRORS.get('assistant', 'unknown error')}")

        registry = ToolRegistry()

        def wiggle(cycles: int = 1, tail: bool = True, body: bool = True) -> str:
            cycles = clamp_int(cycles, 1, 5, 1)
            motion = self.get_motion()
            motion.wiggle(cycles=cycles, tail=bool(tail), body=bool(body))
            return f"wiggle queued for {cycles} cycle(s)"

        def open_mouth(duration: Optional[float] = None, speed: Optional[float] = None) -> str:
            safe_duration = None
            safe_speed = None
            if duration is not None:
                safe_duration = clamp_float(duration, 0.01, 0.5, 0.09)
            if speed is not None:
                safe_speed = clamp_float(speed, 0.0, 100.0, 80.0)
            motion = self.get_motion()
            motion.open_mouth(duration=safe_duration, speed=safe_speed)
            return "mouth open queued"

        def stop_motion() -> str:
            motion = self.get_motion()
            motion.stop_all()
            return "motion stopped"

        def list_voices() -> str:
            tts = self.get_tts()
            voices = tts.available_voices()
            return ", ".join(voices) if voices else "no voices found"

        def set_voice(voice: str) -> str:
            tts = self.get_tts()
            tts.set_voice(str(voice))
            cfg = load_config()
            cfg["tts"]["default_voice"] = str(voice)
            save_config(cfg)
            return f"voice set to {voice}"

        def get_current_time() -> str:
            return time.strftime("%Y-%m-%d %H:%M:%S")

        def set_mode(mode: str) -> str:
            allowed = {"assistant", "bluetooth"}
            if mode not in allowed:
                raise ValueError(f"mode must be one of {sorted(allowed)}")
            cfg = load_config()
            cfg.setdefault("assistant", {})
            cfg["assistant"]["mode"] = mode
            save_config(cfg)
            return f"mode set to {mode}"

        registry.register(Tool("wiggle", "Make the fish wiggle. Args: cycles 1-5, optional tail/body booleans.", wiggle, "safe"))
        registry.register(Tool("open_mouth", "Open the mouth once. Args: optional duration and speed.", open_mouth, "safe"))
        registry.register(Tool("stop_motion", "Stop all fish motion immediately. Args: none.", stop_motion, "safe"))
        registry.register(Tool("get_current_time", "Get current system time. Args: none.", get_current_time, "safe"))
        registry.register(Tool("list_voices", "List available Piper TTS voices. Args: none.", list_voices, "safe"))
        registry.register(Tool("set_voice", "Set current Piper TTS voice. Args: voice string.", set_voice, "safe"))
        registry.register(Tool("set_mode", "Set mode to assistant or bluetooth. Args: mode.", set_mode, "safe"))
        return registry

    def get_assistant(self) -> Any:
        if AssistantConfig is None or AssistantService is None:
            raise RuntimeError(f"assistant_service import failed: {SERVICE_IMPORT_ERRORS.get('assistant', 'unknown error')}")

        with self.lock:
            if self.assistant is not None:
                return self.assistant

            cfg = load_config()["assistant"]
            self.assistant = AssistantService(
                llm=self.get_llm(),
                config=AssistantConfig(
                    assistant_name=str(cfg.get("assistant_name", "Fishseus")),
                    user_name=str(cfg.get("user_name", "Caleb")),
                    personality_path=Path(cfg.get("personality_path", str(PERSONALITY_FILE))),
                    memory_path=Path(cfg.get("memory_path", str(PROJECT_ROOT / "data" / "assistant_memory.json"))),
                    history_path=Path(cfg.get("history_path", str(PROJECT_ROOT / "data" / "conversation_log.jsonl"))),
                    max_history_turns=int(cfg.get("max_history_turns", 8)),
                    max_tool_calls=int(cfg.get("max_tool_calls", 3)),
                    temperature=float(cfg.get("temperature", 0.7)),
                    max_tokens=int(cfg.get("max_tokens", load_config()["llm"].get("max_tokens", 512))),
                    require_explicit_memory_intent=bool(cfg.get("require_explicit_memory_intent", True)),
                ),
                tool_registry=self.build_tool_registry(),
            )
            return self.assistant

    def talk(self, message: str, speak: bool = False, animate: bool = True) -> dict[str, Any]:
        message = message.strip()
        if not message:
            return {"ok": False, "error": "Empty message"}

        try:
            assistant = self.get_assistant()
            result = assistant.handle_user_text(message)

            response: dict[str, Any] = {
                "ok": True,
                "reply": result.speak,
                "motion": result.motion,
                "tool_calls": [call.__dict__ for call in result.tool_calls],
                "tool_results": result.tool_results,
                "memory_updates": result.memory_updates,
                "elapsed_s": result.elapsed_s,
            }

            if speak:
                tts = self.get_tts()
                wav_path = tts.synthesize(result.speak)
                response["tts_wav"] = str(wav_path)

                if animate:
                    try:
                        motion = self.get_motion()
                        motion.speak_audio(wav_path)
                    except Exception as exc:
                        response["motion_error"] = f"{type(exc).__name__}: {exc}"

                tts.play_wav(wav_path, blocking=True)

            return response

        except Exception as exc:
            self.last_error = traceback.format_exc()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": self.last_error}


SERVICES = ServiceManager()


# ---------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------

class FishseusRequestHandler(BaseHTTPRequestHandler):
    server_version = "FishseusWeb/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {fmt % args}")

    def _send_json(self, obj: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            parsed = json.loads(body.decode("utf-8") or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _send_file(self, path: Path) -> None:
        try:
            content = path.read_bytes()
        except Exception:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        suffix = path.suffix.lower()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
        }.get(suffix, "application/octet-stream")

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            ui_file = get_ui_file()
            if ui_file is None:
                self._send_json(
                    {
                        "error": "UI file not found",
                        "searched": [str(p) for p in UI_FILE_CANDIDATES],
                    },
                    HTTPStatus.NOT_FOUND,
                )
                return
            self._send_file(ui_file)
            return

        if path == "/api/status":
            self._send_json(SERVICES.status())
            return

        if path == "/api/config":
            self._send_json(load_config())
            return

        if path == "/api/personality":
            ensure_files()
            self._send_json({"text": PERSONALITY_FILE.read_text(encoding="utf-8")})
            return

        if path == "/api/voices":
            self._send_json({"voices": SERVICES.status().get("voices", [])})
            return

        if path.startswith("/api/"):
            section = path[len("/api/") :]
            config = load_config()
            if section in config:
                self._send_json(config[section])
            else:
                self._send_json({"error": f"Unknown section '{section}'"}, HTTPStatus.NOT_FOUND)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        data = self._read_json()

        try:
            if path == "/api/talk":
                config = load_config()
                speak = bool(data.get("speak", config.get("web", {}).get("enable_talk_tts_by_default", False)))
                animate = bool(data.get("animate", config.get("web", {}).get("enable_talk_motion_by_default", True)))
                result = SERVICES.talk(str(data.get("message", "")), speak=speak, animate=animate)
                status = HTTPStatus.OK if result.get("ok") else HTTPStatus.INTERNAL_SERVER_ERROR
                self._send_json(result, status)
                return

            if path == "/api/personality":
                text = str(data.get("text", ""))
                PERSONALITY_FILE.parent.mkdir(parents=True, exist_ok=True)
                PERSONALITY_FILE.write_text(text, encoding="utf-8")
                SERVICES.reset_assistant()
                self._send_json({"ok": True, "status": "personality saved; assistant will reload on next message"})
                return

            if path == "/api/tts/test":
                text = str(data.get("text", "Behold. Fishseus speaks from the plastic depths."))
                speak = bool(data.get("play", True))
                tts = SERVICES.get_tts()
                wav_path = tts.synthesize(text)
                if speak:
                    tts.play_wav(wav_path, blocking=True)
                self._send_json({"ok": True, "wav_path": str(wav_path)})
                return

            if path == "/api/motor/open_mouth":
                duration = data.get("duration")
                speed = data.get("speed")
                safe_duration = None if duration in (None, "") else clamp_float(duration, 0.01, 0.5, 0.09)
                safe_speed = None if speed in (None, "") else clamp_float(speed, 0.0, 100.0, 80.0)
                motion = SERVICES.get_motion()
                motion.open_mouth(duration=safe_duration, speed=safe_speed)
                self._send_json({"ok": True, "status": "mouth open queued"})
                return

            if path == "/api/motor/wiggle":
                cycles = clamp_int(data.get("cycles", 1), 1, 5, 1)
                tail = bool(data.get("tail", True))
                body = bool(data.get("body", True))
                motion = SERVICES.get_motion()
                motion.wiggle(cycles=cycles, tail=tail, body=body)
                self._send_json({"ok": True, "status": f"wiggle queued for {cycles} cycle(s)"})
                return

            if path == "/api/motor/stop":
                motion = SERVICES.get_motion()
                motion.stop_all()
                self._send_json({"ok": True, "status": "motion stopped"})
                return

            if path == "/api/services/reset":
                SERVICES.shutdown_all()
                self._send_json({"ok": True, "status": "runtime services reset"})
                return

            if path.startswith("/api/"):
                section = path[len("/api/") :]
                config = load_config()

                if section not in config:
                    self._send_json({"error": f"Unknown section '{section}'"}, HTTPStatus.NOT_FOUND)
                    return

                if not isinstance(data, dict):
                    self._send_json({"error": "Invalid payload"}, HTTPStatus.BAD_REQUEST)
                    return

                config[section] = deep_merge(config.get(section, {}), data)
                save_config(config)

                # Rebuild affected runtime services on next use.
                if section in {"llm", "assistant"}:
                    SERVICES.reset_assistant()
                elif section == "motion":
                    SERVICES.reset_motion()
                elif section == "tts":
                    SERVICES.reset_tts()

                self._send_json({"ok": True, "status": f"{section} saved"})
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        except Exception as exc:
            SERVICES.last_error = traceback.format_exc()
            self._send_json(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": SERVICES.last_error,
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def run_server(port: int) -> None:
    ensure_files()

    httpd = ThreadingHTTPServer(("", port), FishseusRequestHandler)

    def handle_signal(signum: int, frame: Any) -> None:
        print("\n[web] stopping...")
        SERVICES.shutdown_all()
        httpd.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"[web] Fishseus control panel: http://0.0.0.0:{port}/")
    print(f"[web] project root: {PROJECT_ROOT}")
    print(f"[web] config: {CONFIG_FILE}")
    if SERVICE_IMPORT_ERRORS:
        print("[web] service import warnings:")
        for name, error in SERVICE_IMPORT_ERRORS.items():
            print(f"  - {name}: {error}")

    try:
        httpd.serve_forever()
    finally:
        SERVICES.shutdown_all()
        httpd.server_close()


if __name__ == "__main__":
    selected_port = 8000
    if len(sys.argv) > 1:
        try:
            selected_port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port {sys.argv[1]!r}; using 8000")
    run_server(selected_port)
