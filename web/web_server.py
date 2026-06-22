#!/usr/bin/env python3
"""
Flask web server for the Fishseus control panel.

Serves the control panel UI and exposes JSON API endpoints to configure
and control the fish assistant services.  Uses real service imports with
graceful degradation — unavailable services (e.g. no RPi.GPIO) are
reported as "unavailable" in the API and their controls are disabled in
the UI.

Usage:
    pip install flask
    python web_server.py [port]

The server binds to 0.0.0.0 on the given port (default 8000).
"""

from __future__ import annotations

import atexit
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "fish_config.json"

# Add each service directory to sys.path so bare module imports work,
# matching the pattern used by the orchestrator.
for _subdir in ("audio", "stt", "llm", "assistant", "motion", "tts"):
    _candidate = PROJECT_ROOT / _subdir
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))


# ---------------------------------------------------------------------
# Service availability — import each independently
# ---------------------------------------------------------------------

_available: dict[str, bool] = {
    "audio": False,
    "llm": False,
    "assistant": False,
    "motion": False,
    "tts": False,
}

# Audio (requires ALSA / arecord — Linux only)
try:
    from audio_service import AudioConfig, AudioService  # type: ignore[import-untyped]
    _available["audio"] = True
except Exception:
    AudioConfig = AudioService = None  # type: ignore[assignment, misc]

# LLM
try:
    from llm_service import LlmConfig, LlmService  # type: ignore[import-untyped]
    _available["llm"] = True
except Exception:
    LlmConfig = LlmService = None  # type: ignore[assignment, misc]

# Assistant (also needs llm to function)
try:
    from assistant_service import AssistantConfig, AssistantService  # type: ignore[import-untyped]
    _available["assistant"] = True
except Exception:
    AssistantConfig = AssistantService = None  # type: ignore[assignment, misc]

# Motion (requires RPi.GPIO — will fail on non-Pi machines)
try:
    from motion_service import MotionService, MotorConfig  # type: ignore[import-untyped]
    _available["motion"] = True
except Exception:
    MotionService = MotorConfig = None  # type: ignore[assignment, misc]

# TTS
try:
    from tts_service import TtsConfig, TtsService  # type: ignore[import-untyped]
    _available["tts"] = True
except Exception:
    TtsConfig = TtsService = None  # type: ignore[assignment, misc]

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
}


# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------

def load_config() -> dict[str, object]:
    """Load the configuration from disk or return defaults."""
    if not CONFIG_FILE.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        for key, val in data.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
                merged[key].update(val)
            else:
                merged[key] = val
        return merged
    except Exception:
        return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(config: dict[str, object]) -> None:
    """Write the configuration to disk."""
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Service instances
# ---------------------------------------------------------------------

_audio: "AudioService | None" = None
_llm: "LlmService | None" = None
_assistant: "AssistantService | None" = None
_motion: "MotionService | None" = None
_tts: "TtsService | None" = None

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
    global _audio, _llm, _assistant, _motion, _tts
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
            _motion = MotionService(
                motors=_build_motor_configs(mc),
                pwm_frequency=int(mc.get("pwm_frequency", 1000)),
                body_wiggle_time=float(mc.get("body_wiggle_time", 0.18)),
                tail_wiggle_time=float(mc.get("tail_wiggle_time", 0.14)),
                mouth_open_time=float(mc.get("mouth_open_time", 0.09)),
                mouth_close_time=float(mc.get("mouth_close_time", 0.04)),
                envelope_window_s=float(mc.get("envelope_window_s", 0.18)),
            )
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
    global _audio, _llm, _assistant, _motion, _tts

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
            _llm.close()
        except Exception:
            pass

    _audio = None
    _llm = None
    _assistant = None
    _motion = None
    _tts = None

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
            _llm.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------

app = Flask(__name__)


# --- Static files ---

@app.route("/")
def index():
    """Serve the control panel HTML."""
    return send_from_directory(str(ROOT_DIR), "fishseus_ui.html")


@app.route("/styles.css")
def serve_css():
    """Serve the stylesheet."""
    return send_from_directory(str(ROOT_DIR), "styles.css", mimetype="text/css")


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
        save_config(json.loads(json.dumps(DEFAULT_CONFIG)))

    # Start services
    init_services()
    atexit.register(shutdown_services)

    unavailable = [k for k, v in _available.items() if not v]
    if unavailable:
        print(f"[web] Unavailable services (graceful degradation): {', '.join(unavailable)}")

    print(f"[web] Serving Fishseus control panel on http://0.0.0.0:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)