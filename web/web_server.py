#!/usr/bin/env python3
"""
Flask web server for the Fishseus control panel.

Serves the control panel UI and exposes JSON API endpoints to configure
and control the fish assistant services.  Uses real service imports with
graceful degradation — unavailable services (e.g. no RPi.GPIO) are
reported as "unavailable" in the API and their controls are disabled in
the UI.

Usage:
    python -m venv web/.venv --system-site-packages
    web/.venv/bin/pip install -r web/requirements.txt
    web/.venv/bin/python web/web_server.py [port]

The server binds to 127.0.0.1 by default (loopback only, intended to sit behind
a Cloudflare Tunnel).  Override with FISHSEUS_BIND_HOST or the web.bind_host
config key (e.g. 0.0.0.0 for LAN access).  See web/DEPLOYMENT.md.
"""

from __future__ import annotations

import atexit
import copy
import json
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

# Access control / hardening lives in a sibling module.  Support both import
# styles: package import (via fishseus.py -> web.web_server) and standalone
# (python web/web_server.py, where web/ is on sys.path[0]).
try:
    from . import security
except ImportError:  # running as a script
    import security  # type: ignore[no-redef]

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "fish_config.json"

# Ensure the repo root is importable so package-style imports resolve when this
# module runs standalone (python web/web_server.py) as well as via fishseus.py.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Service availability — import each independently
# ---------------------------------------------------------------------

_available: dict[str, bool] = {
    "audio": False,
    "llm": False,
    "assistant": False,
    "motion": False,
    "tts": False,
    "vision": False,
    "sensors": False,
}

# Audio (requires ALSA / arecord — Linux only)
try:
    from audio.audio_service import AudioConfig, AudioService  # type: ignore[import-untyped]
    _available["audio"] = True
except Exception:
    AudioConfig = AudioService = None  # type: ignore[assignment, misc]

# LLM
try:
    from llm.llm_service import LlmConfig, LlmService  # type: ignore[import-untyped]
    _available["llm"] = True
except Exception:
    LlmConfig = LlmService = None  # type: ignore[assignment, misc]

# Assistant (also needs llm to function)
try:
    from assistant.assistant_service import AssistantConfig, AssistantService, ToolCall  # type: ignore[import-untyped]
    _available["assistant"] = True
except Exception:
    AssistantConfig = AssistantService = ToolCall = None  # type: ignore[assignment, misc]

# Motion (requires RPi.GPIO — will fail on non-Pi machines)
try:
    from motion.motion_service import MotionConfig, MotionService, MotorConfig  # type: ignore[import-untyped]
    _available["motion"] = True
except Exception:
    MotionService = MotorConfig = None  # type: ignore[assignment, misc]

# TTS
try:
    from tts.tts_service import TtsConfig, TtsService  # type: ignore[import-untyped]
    _available["tts"] = True
except Exception:
    TtsConfig = TtsService = None  # type: ignore[assignment, misc]

# Vision (no hardware deps — just needs requests)
try:
    from vision.vision_service import VisionConfig, VisionService  # type: ignore[import-untyped]
    _available["vision"] = True
except Exception:
    VisionConfig = VisionService = None  # type: ignore[assignment, misc]

# Assistant requires a working LLM
if _available["assistant"] and not _available["llm"]:
    _available["assistant"] = False


# ---------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, object] = {
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
        "whisper_binary": "../stt/whisper.cpp/build/bin/whisper-cli",
        "model_path": "../stt/whisper.cpp/models/ggml-base.en.bin",
        "threads": 4,
        "wake_words": ["fish", "fishseus", "hey fish"],
    },
    "llm": {
        "endpoint_url": "http://ollama.angelfish-gamma.ts.net/v1/chat/completions",
        "model": "qwen2.5:3b",
        "temperature": 0.7,
        "max_tokens": 512,
        "disable_reasoning": True,
    },
    "tts": {
        "piper_binary": "../tts/.venv/bin/piper",
        "voices_dir": "../tts/voices",
        "default_voice": "en_US-arctic-medium",
        "audio_device": "plughw:0,0",
        "output_dir": "../tmp/tts",
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
                "in1": 17, "in2": 27, "en": 22,
                "forward_speed": 82, "reverse_speed": 55,
                "neutral_return_time": 0.04,
            },
            "tail": {
                "in1": 23, "in2": 24, "en": 25,
                "forward_speed": 72, "reverse_speed": 48,
                "neutral_return_time": 0.03,
            },
            "body": {
                "in1": 5, "in2": 6, "en": 12,
                "forward_speed": 68, "reverse_speed": 45,
                "neutral_return_time": 0.03,
            },
        },
    },
    "assistant": {
        "assistant_name": "Fishseus",
        "user_name": "User",
        "personality_path": "../config/personality_prompt.txt",
    },
    "vision": {
        "endpoint_url": "http://ollama.angelfish-gamma.ts.net/v1/chat/completions",
        "model": "qwen2.5vl:3b",
        "timeout_s": 60,
        "max_tokens": 300,
        "default_camera": "front",
        "cameras": {},
    },
    "sensors": {
        "enabled": False,
        "poll_interval_s": 0.05,
        "sensors": [],
    },
    # Web server / access control.  Secrets (local_bypass_token) are better set
    # via environment variables (FISHSEUS_*) since this file is committed to git;
    # see web/security.py.  These are the non-secret defaults.
    "web": {
        "bind_host": "127.0.0.1",
        "auth_enabled": True,
        "cf_access_team_domain": "",
        "cf_access_aud": "",
        "local_bypass_token": "",
        "devices_cache_ttl_s": 60.0,
    },
}


# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------

# Config is read on nearly every request (the amplitude graph polls at ~2 Hz),
# so we cache the parsed+merged result and only re-read when the file's mtime
# changes.  This removes repeated disk I/O and JSON parsing from the hot path.
_config_cache: dict[str, object] | None = None
_config_mtime: float | None = None
_config_lock = threading.Lock()


def _merge_defaults(data: dict) -> dict[str, object]:
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for key, val in data.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key].update(val)
        else:
            merged[key] = val
    return merged


def load_config() -> dict[str, object]:
    """Load the configuration (cached, invalidated on file mtime change).

    Returns a deep copy so callers may freely mutate the result without
    corrupting the cache.
    """
    global _config_cache, _config_mtime

    with _config_lock:
        if not CONFIG_FILE.exists():
            return copy.deepcopy(DEFAULT_CONFIG)
        try:
            mtime = CONFIG_FILE.stat().st_mtime
            if _config_cache is None or mtime != _config_mtime:
                with CONFIG_FILE.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                _config_cache = _merge_defaults(data)
                _config_mtime = mtime
            return copy.deepcopy(_config_cache)
        except Exception:
            return copy.deepcopy(DEFAULT_CONFIG)


def save_config(config: dict[str, object]) -> None:
    """Write the configuration to disk and refresh the in-memory cache."""
    global _config_cache, _config_mtime
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    with _config_lock:
        _config_cache = copy.deepcopy(config)
        try:
            _config_mtime = CONFIG_FILE.stat().st_mtime
        except Exception:
            _config_mtime = None


# ---------------------------------------------------------------------
# Service instances
# ---------------------------------------------------------------------

_audio: "AudioService | None" = None
_llm: "LlmService | None" = None
_assistant: "AssistantService | None" = None
_motion: "MotionService | None" = None
_tts: "TtsService | None" = None
_vision: "VisionService | None" = None
_sensors = None  # SensorService — only ever injected by the orchestrator
_tool_registry = None  # injected by orchestrator via set_tool_registry()

# Rolling amplitude buffer for the live graph
_AMP_HISTORY_SIZE = 200  # ~20 seconds at 10 Hz
_amp_history: deque[dict] = deque(maxlen=_AMP_HISTORY_SIZE)
_amp_lock = threading.Lock()


def _build_motor_configs(motion_conf: dict) -> "dict[str, MotorConfig]":
    """Build MotorConfig dataclass instances from config dict."""
    motors_raw = motion_conf.get("motors", {})
    motors = {}
    for name, m in motors_raw.items():
        motors[name] = MotorConfig(
            in1=int(m.get("in1", 0)),
            in2=int(m.get("in2", 0)),
            en=int(m.get("en", 0)),
            forward_speed=float(m.get("forward_speed", 70)),
            reverse_speed=float(m.get("reverse_speed", 55)),
            neutral_return_time=float(m.get("neutral_return_time", 0.08)),
        )
    return motors


def init_services() -> None:
    """Initialise all available services using current config."""
    global _audio, _llm, _assistant, _motion, _tts, _vision
    config = load_config()

    # --- Audio ---
    if _available["audio"] and _audio is None:
        try:
            ac = config.get("audio", {})
            _audio = AudioService(
                AudioConfig(
                    device=ac.get("device", "default"),
                    sample_rate=int(ac.get("sample_rate", 16000)),
                    channels=int(ac.get("channels", 1)),
                    speech_threshold=float(ac.get("speech_threshold", 0.003)),
                    silence_threshold=float(ac.get("silence_threshold", 0.0008)),
                    silence_timeout_s=float(ac.get("silence_timeout_s", 1.0)),
                    max_record_seconds=float(ac.get("max_record_seconds", 12.0)),
                    pre_roll_seconds=float(ac.get("pre_roll_seconds", 0.4)),
                )
            )
            _audio.initialize()
            print("[web] Audio service ready")
        except Exception as exc:
            print(f"[web] Audio service init failed: {exc}")
            _audio = None
            _available["audio"] = False

    # --- Motion ---
    if _available["motion"] and _motion is None:
        try:
            mc = config.get("motion", {})
            motors = _build_motor_configs(mc)
            cfg_kwargs = dict(
                pwm_frequency=int(mc.get("pwm_frequency", 1000)),
                body_wiggle_time=float(mc.get("body_wiggle_time", 0.18)),
                tail_wiggle_time=float(mc.get("tail_wiggle_time", 0.14)),
                mouth_open_time=float(mc.get("mouth_open_time", 0.09)),
                mouth_close_time=float(mc.get("mouth_close_time", 0.04)),
                envelope_window_s=float(mc.get("envelope_window_s", 0.18)),
            )
            if motors:
                cfg_kwargs["motors"] = motors  # else MotionConfig's default pinout
            _motion = MotionService(MotionConfig(**cfg_kwargs))
            _motion.initialize()
            print("[web] Motion service ready")
        except Exception as exc:
            print(f"[web] Motion service init failed: {exc}")
            _motion = None
            _available["motion"] = False

    # --- LLM ---
    if _available["llm"] and _llm is None:
        try:
            lc = config.get("llm", {})
            _llm = LlmService(
                LlmConfig(
                    endpoint_url=lc.get("endpoint_url"),
                    model=lc.get("model"),
                    timeout_s=60.0,
                    retries=1,
                    temperature=float(lc.get("temperature", 0.7)),
                    max_tokens=int(lc.get("max_tokens", 512)),
                    disable_reasoning=bool(lc.get("disable_reasoning", True)),
                )
            )
            print("[web] LLM service ready")
        except Exception as exc:
            print(f"[web] LLM service init failed: {exc}")
            _llm = None
            _available["llm"] = False

    # --- Assistant (requires LLM) ---
    if _available["assistant"] and _assistant is None and _llm is not None:
        try:
            ac = config.get("assistant", {})
            _assistant = AssistantService(
                llm=_llm,
                config=AssistantConfig(
                    assistant_name=ac.get("assistant_name", "Fishseus"),
                    user_name=ac.get("user_name", "User"),
                    personality_path=Path(
                        ac.get("personality_path", "../config/personality_prompt.txt")
                    ).resolve(),
                    memory_path=Path(
                        ac.get(
                            "memory_path",
                            str(PROJECT_ROOT / "data" / "assistant_memory.json"),
                        )
                    ),
                    history_path=Path(
                        ac.get(
                            "history_path",
                            str(PROJECT_ROOT / "data" / "conversation_log.jsonl"),
                        )
                    ),
                ),
            )
            print("[web] Assistant service ready")
        except Exception as exc:
            print(f"[web] Assistant service init failed: {exc}")
            _assistant = None
            _available["assistant"] = False

    # --- Vision ---
    if _available["vision"] and _vision is None:
        try:
            vc = config.get("vision", {})
            if vc.get("cameras"):
                _vision = VisionService(
                    VisionConfig(
                        endpoint_url=vc.get("endpoint_url", ""),
                        model=vc.get("model", "qwen2.5vl:3b"),
                        timeout_s=float(vc.get("timeout_s", 60.0)),
                        max_tokens=int(vc.get("max_tokens", 300)),
                        capture_dir=PROJECT_ROOT / "tmp" / "vision",
                        cameras=vc.get("cameras", {}),
                        default_camera=vc.get("default_camera", ""),
                    )
                )
                print("[web] Vision service ready")
            else:
                _available["vision"] = False
        except Exception as exc:
            print(f"[web] Vision service init failed: {exc}")
            _vision = None
            _available["vision"] = False

    # --- TTS ---
    if _available["tts"] and _tts is None:
        try:
            tc = config.get("tts", {})
            _tts = TtsService(
                TtsConfig(
                    piper_binary=tc.get("piper_binary", "piper"),
                    voices_dir=Path(
                        tc.get("voices_dir", str(PROJECT_ROOT / "tts" / "voices"))
                    ),
                    default_voice=tc.get("default_voice", "en_US-arctic-medium"),
                    audio_device=tc.get("audio_device", "plughw:0,0"),
                    output_dir=Path(
                        tc.get("output_dir", str(PROJECT_ROOT / "tmp" / "tts"))
                    ),
                )
            )
            print("[web] TTS service ready")
        except Exception as exc:
            print(f"[web] TTS service init failed: {exc}")
            _tts = None
            _available["tts"] = False


def reset_services() -> None:
    """Tear down all service instances so they can be reinitialised."""
    global _audio, _llm, _assistant, _motion, _tts, _vision

    if _audio is not None:
        try:
            _audio.shutdown()
        except Exception:
            pass

    if _motion is not None:
        try:
            _motion.stop_all()
            _motion.shutdown()
        except Exception:
            pass

    if _llm is not None:
        try:
            _llm.shutdown()
        except Exception:
            pass

    if _vision is not None:
        try:
            _vision.shutdown()
        except Exception:
            pass

    _audio = None
    _llm = None
    _assistant = None
    _motion = None
    _tts = None
    _vision = None

    with _amp_lock:
        _amp_history.clear()

    # Reset availability flags to allow re-detection
    for key in _available:
        _available[key] = True

    # Re-check what's importable
    _available["audio"] = AudioService is not None
    _available["llm"] = LlmService is not None
    _available["assistant"] = AssistantService is not None and LlmService is not None
    _available["motion"] = MotionService is not None
    _available["tts"] = TtsService is not None
    _available["vision"] = VisionService is not None
    _available["sensors"] = False  # only ever available via orchestrator injection


def set_tool_registry(registry) -> None:
    """
    Inject the shared ToolRegistry from the orchestrator.

    Called by the orchestrator after tool_registry is built so the web API
    controls the same registry the assistant is using.
    """
    global _tool_registry
    _tool_registry = registry


def set_services(
    audio=None,
    llm=None,
    assistant=None,
    motion=None,
    tts=None,
    vision=None,
    sensors=None,
) -> None:
    """
    Inject pre-initialised service instances from an external orchestrator.

    Call this before starting the Flask thread so the web UI controls the
    same service objects the orchestrator is using.  Any argument left as
    None is left unchanged.
    """
    global _audio, _llm, _assistant, _motion, _tts, _vision, _sensors

    if audio is not None:
        _audio = audio
        _available["audio"] = True
    if llm is not None:
        _llm = llm
        _available["llm"] = True
    if assistant is not None:
        _assistant = assistant
        _available["assistant"] = True
    if motion is not None:
        _motion = motion
        _available["motion"] = True
    if tts is not None:
        _tts = tts
        _available["tts"] = True
    if vision is not None:
        _vision = vision
        _available["vision"] = True
    if sensors is not None:
        _sensors = sensors
        _available["sensors"] = True


def shutdown_services() -> None:
    """Clean shutdown of services for atexit."""
    if _audio is not None:
        try:
            _audio.shutdown()
        except Exception:
            pass
    if _motion is not None:
        try:
            _motion.stop_all()
            _motion.shutdown()
        except Exception:
            pass
    if _llm is not None:
        try:
            _llm.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------

app = Flask(__name__)

# Reject oversized request bodies outright (defends the JSON parsers).
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MiB


# --- Access control & hardening (see web/security.py) ---

# Endpoints reachable without authentication.  Keep this minimal: /healthz is
# for tunnel / uptime probes and leaks nothing.
_AUTH_EXEMPT = {"/healthz"}


@app.before_request
def _enforce_auth():
    if request.method == "OPTIONS" or request.path in _AUTH_EXEMPT:
        return None
    if not security.csrf_ok(request):
        return jsonify({"error": "Cross-origin request rejected"}), 403
    web_conf = load_config().get("web", {})
    allowed, message, status = security.authorize(request, web_conf)
    if not allowed:
        return jsonify({"error": message}), status
    return None


@app.after_request
def _add_security_headers(response):
    return security.apply_security_headers(response)


@app.route("/healthz")
def healthz():
    """Unauthenticated liveness probe for the tunnel / uptime monitors."""
    return jsonify({"status": "ok"})


# --- Static files ---

@app.route("/")
def index():
    """Serve the control panel HTML."""
    return send_from_directory(str(ROOT_DIR), "fishseus_ui.html")


@app.route("/styles.css")
def serve_css():
    """Serve the stylesheet."""
    return send_from_directory(
        str(ROOT_DIR), "styles.css", mimetype="text/css", max_age=3600
    )


# --- Status ---

@app.route("/api/status")
def api_status():
    """Return real-time service availability."""
    return jsonify(
        {
            "services": {
                "audio": "ready" if _available["audio"] and _audio is not None else "unavailable",
                "llm": "ready" if _available["llm"] and _llm is not None else "unavailable",
                "assistant": "ready" if _available["assistant"] and _assistant is not None else "unavailable",
                "motion": "ready" if _available["motion"] and _motion is not None else "unavailable",
                "tts": "ready" if _available["tts"] and _tts is not None else "unavailable",
                "vision": "ready" if _available["vision"] and _vision is not None else "unavailable",
                "sensors": "ready" if _available["sensors"] and _sensors is not None else "unavailable",
            }
        }
    )


# --- Configuration ---

@app.route("/api/config")
def api_config():
    """Return the full configuration."""
    return jsonify(load_config())


@app.route("/api/<section>", methods=["GET", "POST"])
def api_section(section: str):
    """Get or update a configuration section (audio, stt, llm, tts, motion, assistant)."""
    config = load_config()

    if request.method == "GET":
        if section in config:
            return jsonify(config[section])
        return jsonify({"error": f"Unknown section '{section}'"}), 404

    # POST — update section
    data = request.get_json(silent=True) or {}
    if section not in config:
        return jsonify({"error": f"Unknown section '{section}'"}), 404

    config_section = config.get(section, {})
    if not isinstance(config_section, dict):
        return jsonify({"error": "Cannot update this section"}), 400

    config_section.update(data)
    config[section] = config_section
    save_config(config)
    return jsonify({"status": "saved"})

# --- Hardware device discovery ---

_ALSA_CARD_RE = re.compile(
    r"^card (\d+): (\S+) \[(.+?)\], device (\d+): (.+?) \[(.+?)\]", re.MULTILINE
)


def _list_alsa_devices(command: str) -> list[dict]:
    """Parse `arecord -l` / `aplay -l` output into device entries."""
    try:
        proc = subprocess.run([command, "-l"], capture_output=True, text=True, timeout=5)
    except Exception:
        return []
    if proc.returncode != 0:
        return []

    devices = []
    for m in _ALSA_CARD_RE.finditer(proc.stdout):
        card, card_id, card_name, dev, dev_id, dev_name = m.groups()
        devices.append({
            "id": f"plughw:{card},{dev}",
            "card": int(card),
            "device": int(dev),
            "label": f"{card_name} — {dev_name} (plughw:{card},{dev})",
        })
    return devices


def _list_cameras() -> list[dict]:
    """Detect cameras: rpicam CSI cameras and /dev/video* V4L2 devices."""
    cameras = []

    # CSI cameras via rpicam-hello
    try:
        proc = subprocess.run(
            ["rpicam-hello", "--list-cameras"],
            capture_output=True, text=True, timeout=8,
        )
        if proc.returncode == 0:
            for m in re.finditer(r"^(\d+)\s*:\s*(\S+)\s*\[(.+?)\]", proc.stdout, re.MULTILINE):
                idx, sensor, modes = m.groups()
                cameras.append({
                    "id": f"csi:{idx}",
                    "label": f"CSI camera {idx}: {sensor} [{modes}]",
                    "suggested_command": (
                        f"rpicam-still --camera {idx} -o {{output}} "
                        "--width 1280 --height 720 --nopreview -t 500"
                    ),
                })
    except Exception:
        pass

    # USB / V4L2 devices
    try:
        for dev in sorted(Path("/dev").glob("video*")):
            name = ""
            name_file = Path(f"/sys/class/video4linux/{dev.name}/name")
            if name_file.exists():
                name = name_file.read_text().strip()
            cameras.append({
                "id": str(dev),
                "label": f"{dev} — {name}" if name else str(dev),
                "suggested_command": f"fswebcam -d {dev} -r 1280x720 --no-banner {{output}}",
            })
    except Exception:
        pass

    return cameras


# Device discovery spawns subprocesses (arecord/aplay -l, rpicam-hello) which is
# expensive on a Pi.  The UI calls this on every page load, so cache the result
# for a short TTL; hardware rarely changes between page loads.  Pass ?refresh=1
# to force a rescan.
_devices_cache: dict | None = None
_devices_cache_ts: float = 0.0
_devices_lock = threading.Lock()


@app.route("/api/devices")
def api_devices():
    """List available capture, playback, and camera devices on this machine."""
    global _devices_cache, _devices_cache_ts

    ttl = float(load_config().get("web", {}).get("devices_cache_ttl_s", 60.0))
    force = request.args.get("refresh") in ("1", "true", "yes")

    with _devices_lock:
        fresh = (
            _devices_cache is not None
            and not force
            and (time.time() - _devices_cache_ts) < ttl
        )
        if fresh:
            return jsonify(_devices_cache)

    # Do the (slow) discovery outside the lock so concurrent callers don't block.
    result = {
        "capture": _list_alsa_devices("arecord"),
        "playback": _list_alsa_devices("aplay"),
        "cameras": _list_cameras(),
    }
    with _devices_lock:
        _devices_cache = result
        _devices_cache_ts = time.time()
    return jsonify(result)


# --- Audio amplitude streaming ---

def _amplitude_callback(amp: float) -> None:
    """Called by the audio capture thread; stash amplitude in the ring buffer."""
    entry = {"t": round(time.time(), 3), "a": round(amp, 6)}
    with _amp_lock:
        _amp_history.append(entry)


@app.route("/api/audio/amplitude")
def api_audio_amplitude():
    """Return the rolling amplitude history and the current threshold values."""
    config = load_config()
    ac = config.get("audio", {})
    with _amp_lock:
        samples = list(_amp_history)
    return jsonify({
        "available": _available.get("audio", False) and _audio is not None,
        "capturing": _audio is not None and _audio._capture_thread is not None
                     and _audio._capture_thread.is_alive() if _audio else False,
        "speech_threshold": float(ac.get("speech_threshold", 0.003)),
        "silence_threshold": float(ac.get("silence_threshold", 0.0008)),
        "samples": samples,
    })


@app.route("/api/audio/capture/start", methods=["POST"])
def api_audio_capture_start():
    """Start the audio capture thread and begin streaming amplitude."""
    if _audio is None:
        return jsonify({"error": "Audio service unavailable"}), 503
    try:
        _audio.start_capture(amplitude_callback=_amplitude_callback)
        return jsonify({"status": "Capture started"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/audio/capture/stop", methods=["POST"])
def api_audio_capture_stop():
    """Stop the audio capture thread."""
    if _audio is None:
        return jsonify({"error": "Audio service unavailable"}), 503
    try:
        _audio.stop_capture()
        with _amp_lock:
            _amp_history.clear()
        return jsonify({"status": "Capture stopped"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# --- Personality prompt (separate file) ---

@app.route("/api/personality", methods=["GET", "POST"])
def api_personality():
    """Read or write the personality prompt text file."""
    config = load_config()
    rel_path = config.get("assistant", {}).get("personality_path", "../config/personality_prompt.txt")
    personality_file = (ROOT_DIR / rel_path).resolve()

    if request.method == "GET":
        try:
            text = personality_file.read_text(encoding="utf-8") if personality_file.exists() else ""
        except Exception:
            text = ""
        return jsonify({"text": text})

    # POST — save
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    try:
        personality_file.parent.mkdir(parents=True, exist_ok=True)
        personality_file.write_text(text, encoding="utf-8")
        return jsonify({"status": "saved"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# --- Talk (assistant chat) ---

@app.route("/api/talk", methods=["POST"])
def api_talk():
    """Send a message to the assistant and receive a reply."""
    if _assistant is None:
        return jsonify({"error": "Assistant service unavailable"}), 503

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        result = _assistant.handle_user_text(message)
        return jsonify(
            {
                "reply": getattr(result, "speak", ""),
                "motion": getattr(result, "motion", "speaking"),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# --- Motor commands ---

@app.route("/api/motor/<command>", methods=["POST"])
def api_motor(command: str):
    """Execute a motor command on the motion service."""
    if _motion is None:
        return jsonify({"error": "Motion service unavailable"}), 503

    data = request.get_json(silent=True) or {}

    try:
        if command == "open_mouth":
            duration = data.get("duration")
            speed = data.get("speed")
            dur_f = float(duration) if duration is not None else None
            spd_f = float(speed) if speed is not None else None
            _motion.open_mouth(duration=dur_f, speed=spd_f)
            return jsonify({"status": "Mouth open queued"})

        if command == "wiggle":
            cycles = int(data.get("cycles", 2))
            tail = bool(data.get("tail", True))
            body = bool(data.get("body", True))
            _motion.wiggle(cycles=cycles, tail=tail, body=body)
            return jsonify({"status": f"Wiggle queued ({cycles} cycles)"})

        if command == "speak_placeholder":
            duration_s = float(data.get("duration_s", 2.0))
            _motion.speak_text_placeholder(duration_s=duration_s)
            return jsonify({"status": f"Speak placeholder queued ({duration_s}s)"})

        if command == "stop":
            _motion.stop_all()
            return jsonify({"status": "All motors stopped"})

        if command == "direct_drive":
            motor = data.get("motor", "")
            direction = data.get("direction", "forward")
            speed = float(data.get("speed", 50))
            _motion.direct_drive(motor, direction, speed)
            return jsonify({"status": f"{motor} driving {direction} at {speed}%"})

        if command == "direct_stop":
            motor = data.get("motor")  # None = all
            _motion.direct_stop(motor)
            return jsonify({"status": f"{'All motors' if motor is None else motor} stopped"})

        return jsonify({"error": f"Unknown motor command '{command}'"}), 404

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# --- Vision (camera view + describe) ---

@app.route("/api/vision/snapshot")
def api_vision_snapshot():
    """Capture a frame from the named camera and return the JPEG."""
    if _vision is None:
        return jsonify({"error": "Vision service unavailable"}), 503
    camera = request.args.get("camera", "")
    try:
        frame = _vision.capture(camera)
        return send_file(str(frame), mimetype="image/jpeg", max_age=0)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/vision/describe", methods=["POST"])
def api_vision_describe():
    """Capture a frame and ask the vision model about it."""
    if _vision is None:
        return jsonify({"error": "Vision service unavailable"}), 503
    data = request.get_json(silent=True) or {}
    camera = str(data.get("camera", ""))
    question = str(data.get("question", "")).strip() or "Describe what you see in one or two sentences."
    try:
        frame = _vision.capture(camera)
        description = _vision.describe(frame, question)
        return jsonify({"description": description, "camera": camera or "default"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/sensors/status")
def api_sensors_status():
    """Live state of every configured sensor (orchestrator mode only)."""
    if _sensors is None:
        return jsonify({"available": False, "sensors": []})
    try:
        return jsonify({"available": True, "sensors": _sensors.sensor_report()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# --- Tool management ---

@app.route("/api/tools", methods=["GET"])
def api_tools_list():
    """List all registered tools with their current state."""
    if _tool_registry is None:
        return jsonify({"tools": []})
    return jsonify({"tools": _tool_registry.list_all()})


@app.route("/api/tools/<name>/enable", methods=["POST"])
def api_tool_enable(name: str):
    """Enable a tool so the LLM can see and call it."""
    if _tool_registry is None:
        return jsonify({"error": "Tool registry unavailable"}), 503
    _tool_registry.enable(name)
    return jsonify({"status": f"{name} enabled"})


@app.route("/api/tools/<name>/disable", methods=["POST"])
def api_tool_disable(name: str):
    """Disable a tool — hides it from the LLM and blocks execution."""
    if _tool_registry is None:
        return jsonify({"error": "Tool registry unavailable"}), 503
    _tool_registry.disable(name)
    return jsonify({"status": f"{name} disabled"})


@app.route("/api/tools/<name>/run", methods=["POST"])
def api_tool_run(name: str):
    """Execute a tool directly, bypassing the LLM."""
    if _tool_registry is None:
        return jsonify({"error": "Tool registry unavailable"}), 503
    if ToolCall is None:
        return jsonify({"error": "assistant_service not importable"}), 503
    data = request.get_json(silent=True) or {}
    args = data.get("args", {})
    try:
        call = ToolCall(name=name, args=args)
        result = _tool_registry.execute(call)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/tools/<name>", methods=["PATCH"])
def api_tool_update(name: str):
    """Edit a tool's description or synthesize_result flag."""
    if _tool_registry is None:
        return jsonify({"error": "Tool registry unavailable"}), 503
    data = request.get_json(silent=True) or {}
    if "description" in data:
        _tool_registry.update_description(name, str(data["description"]))
    if "synthesize_result" in data:
        _tool_registry.set_synthesize_result(name, bool(data["synthesize_result"]))
    return jsonify({"status": "updated"})


# --- Memory and conversation history ---

@app.route("/api/memory", methods=["GET"])
def api_memory():
    """Return the full assistant memory JSON."""
    if _assistant is not None:
        return jsonify(_assistant.memory.data)
    mem_file = PROJECT_ROOT / "data" / "assistant_memory.json"
    if mem_file.exists():
        try:
            return jsonify(json.loads(mem_file.read_text(encoding="utf-8")))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
    return jsonify({"error": "Memory unavailable"}), 503


@app.route("/api/memory/update", methods=["POST"])
def api_memory_update():
    """Update a dotted-key path in memory (e.g. profile.user_name)."""
    if _assistant is None:
        return jsonify({"error": "Assistant unavailable"}), 503
    data = request.get_json(silent=True) or {}
    key = str(data.get("key", "")).strip()
    value = data.get("value")
    if not key:
        return jsonify({"error": "key is required"}), 400
    _assistant.memory.update_path(key, value)
    _assistant.memory.save()
    return jsonify({"status": "updated"})


@app.route("/api/memory/facts", methods=["DELETE"])
def api_memory_clear_facts():
    """Clear all stored facts from memory."""
    if _assistant is None:
        return jsonify({"error": "Assistant unavailable"}), 503
    _assistant.memory.data["facts"] = []
    _assistant.memory.save()
    return jsonify({"status": "facts cleared"})


@app.route("/api/memory/history", methods=["GET"])
def api_memory_history():
    """Return recent conversation log entries."""
    n = request.args.get("n", 30, type=int)
    hist_file = PROJECT_ROOT / "data" / "conversation_log.jsonl"
    if not hist_file.exists():
        return jsonify({"turns": []})
    try:
        raw = hist_file.read_text(encoding="utf-8").strip()
        lines = [l for l in raw.split("\n") if l.strip()]
        recent = lines[-n:]
        turns = []
        for line in recent:
            try:
                turns.append(json.loads(line))
            except Exception:
                pass
        return jsonify({"turns": turns})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/memory/history", methods=["DELETE"])
def api_memory_clear_history():
    """Clear the conversation log file and in-memory history."""
    hist_file = PROJECT_ROOT / "data" / "conversation_log.jsonl"
    try:
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        hist_file.write_text("", encoding="utf-8")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    if _assistant is not None:
        _assistant.history.clear()
    return jsonify({"status": "history cleared"})


# --- Services reset ---

@app.route("/api/services/reset", methods=["POST"])
def api_services_reset():
    """Tear down and reinitialise all services."""
    reset_services()
    init_services()
    return jsonify({"status": "Services reinitialised"})


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port '{sys.argv[1]}', using default 8000")

    # Ensure a config file exists on disk
    if not CONFIG_FILE.exists():
        save_config(copy.deepcopy(DEFAULT_CONFIG))

    # Start services
    init_services()
    atexit.register(shutdown_services)

    unavailable = [k for k, v in _available.items() if not v]
    if unavailable:
        print(f"[web] Unavailable services (graceful degradation): {', '.join(unavailable)}")

    # Bind loopback-only by default; cloudflared reaches us locally.  Override
    # with FISHSEUS_BIND_HOST or web.bind_host (e.g. 0.0.0.0 for LAN access).
    web_conf = load_config().get("web", {})
    host = security.resolve_settings(web_conf)["bind_host"]
    for line in security.startup_report(web_conf):
        print(line)

    print(f"[web] Serving Fishseus control panel on http://{host}:{port}/")
    app.run(host=host, port=port, debug=False, threaded=True)