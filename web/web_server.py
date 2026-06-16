#!/usr/bin/env python3
"""
A minimal web server for the Fishseus control panel.

This server uses Python's built‑in http.server module to avoid external
dependencies.  It serves the HTML UI and exposes simple JSON endpoints
to read and update configuration files.  It also includes mock
endpoints for motor control and a talk feature.

Configuration values are stored in a JSON file (fish_config.json) in
the same directory as this script.  If the file does not exist, it
will be created with reasonable defaults.

Usage:
    python3 web_server.py 8000

The server will bind to all interfaces on the specified port (default
8000).  Navigate to http://localhost:8000/ in a browser to use the
control panel.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
from pathlib import Path


# ---------------------------------------------------------------------
# Configuration and stubs
# ---------------------------------------------------------------------

# Location of the UI HTML file, stylesheet, and the config file.  We use the
# directory of this script as the base.
ROOT_DIR = Path(__file__).resolve().parent
UI_FILE = ROOT_DIR / "fishseus_ui.html"
CSS_FILE = ROOT_DIR / "styles.css"
CONFIG_FILE = ROOT_DIR / "fish_config.json"

# Import the real services if available.  Fall back to stubs when necessary.
try:
    # Try to import the real assistant service.  It should provide an
    # AssistantService class with a handle_user_text method.
    from assistant_service import AssistantService, AssistantConfig
    from llm.llm_service import LlmService, LlmConfig
    from motion_service import MotionService, MotionConfig as RealMotionConfig
    from tts.tts_service import TtsService, TtsConfig

    # Lazy initialization placeholders.  These instances will be created
    # on first use in the API handlers to avoid startup delays and to
    # respect current configuration.
    _assistant_service: AssistantService | None = None
    _motion_service: MotionService | None = None
    _tts_service: TtsService | None = None

    def _init_services(config: dict[str, object]) -> None:
        """Initialise the assistant, motion and tts services using the provided config."""
        global _assistant_service, _motion_service, _tts_service
        # Only initialise once or when forced to reinitialise
        if _assistant_service is None:
            # Build LLM service
            llm_conf = config.get("llm", {})
            llm_service = LlmService(
                LlmConfig(
                    endpoint_url=llm_conf.get("endpoint_url"),
                    model=llm_conf.get("model"),
                    timeout_s=60.0,
                    retries=1,
                    temperature=llm_conf.get("temperature", 0.7),
                    max_tokens=llm_conf.get("max_tokens", 256),
                    disable_reasoning=llm_conf.get("disable_reasoning", True),
                )
            )
            # Create assistant service
            assistant_conf = config.get("assistant", {})
            # Use default assistant and memory paths from config; fallback values are defined in config file.
            acfg = AssistantConfig(
                assistant_name=assistant_conf.get("assistant_name", "Fishseus"),
                user_name=assistant_conf.get("user_name", "User"),
                personality_path=Path(assistant_conf.get("personality_path", "../config/personality_prompt.txt")),
                memory_path=Path(assistant_conf.get("memory_path", "../data/assistant_memory.json")),
                history_path=Path(assistant_conf.get("history_path", "../data/conversation_log.jsonl")),
            )
            _assistant_service = AssistantService(
                config=acfg,
                llm_service=llm_service,
            )
        if _motion_service is None:
            motion_conf = config.get("motion", {})
            # Build real MotionConfig if available; otherwise fallback to stub config
            try:
                mcfg = RealMotionConfig(
                    pwm_frequency=motion_conf.get("pwm_frequency", 1000),
                    body_wiggle_time=motion_conf.get("body_wiggle_time", 0.18),
                    tail_wiggle_time=motion_conf.get("tail_wiggle_time", 0.14),
                    mouth_open_time=motion_conf.get("mouth_open_time", 0.09),
                    mouth_close_time=motion_conf.get("mouth_close_time", 0.04),
                    envelope_window_s=motion_conf.get("envelope_window_s", 0.18),
                    motors=motion_conf.get("motors", {}),
                )
            except Exception:
                # fallback: pass through unknown keys to RealMotionConfig may fail; use default constructor
                mcfg = RealMotionConfig()
            _motion_service = MotionService(mcfg)
        if _tts_service is None:
            tts_conf = config.get("tts", {})
            _tts_service = TtsService(
                TtsConfig(
                    piper_binary=tts_conf.get("piper_binary"),
                    voices_dir=tts_conf.get("voices_dir"),
                    default_voice=tts_conf.get("default_voice"),
                    audio_device=tts_conf.get("audio_device"),
                    output_dir=tts_conf.get("output_dir", "../tmp/tts"),
                )
            )

    def assistant_handle_user_text(message: str) -> dict[str, str]:
        # Ensure services are initialised with latest config
        config = load_config()
        _init_services(config)
        assert _assistant_service is not None
        result = _assistant_service.handle_user_text(message)
        # result is likely an AssistantResult; convert to dict
        return {
            "reply": result.speak if hasattr(result, "speak") else getattr(result, "text", ""),
        }

    def motion_open_mouth(duration: float | None = None, speed: int | None = None) -> str:
        config = load_config()
        _init_services(config)
        assert _motion_service is not None
        try:
            return _motion_service.open_mouth(duration=duration, speed=speed)
        except TypeError:
            # If real service does not support params, fall back
            return _motion_service.open_mouth()  # type: ignore

    def motion_wiggle(cycles: int = 1) -> str:
        config = load_config()
        _init_services(config)
        assert _motion_service is not None
        return _motion_service.wiggle(cycles)

    def reset_services() -> str:
        # reset internal instances so they are reinitialised on next call
        global _assistant_service, _motion_service, _tts_service
        _assistant_service = None
        _motion_service = None
        _tts_service = None
        return "Services reset"

except Exception:
    # Fallback to stubs if real modules cannot be imported
    from assistant_service_stub import handle_user_text as assistant_handle_user_text  # type: ignore
    from motion_service_stub import MotionServiceStub, MotionConfig
    _motion_service = MotionServiceStub()

    def motion_open_mouth(duration: float | None = None, speed: int | None = None) -> str:
        # Use stub which may not accept params; ignoring overrides
        try:
            return _motion_service.open_mouth(duration=duration, speed=speed)  # type: ignore
        except Exception:
            return _motion_service.open_mouth()  # type: ignore

    def motion_wiggle(cycles: int = 1) -> str:
        return _motion_service.wiggle(cycles)

    def reset_services() -> str:
        return "Stub services reset"



# Default configuration values.  These correspond to the defaults used
# in the orchestrator code from the Fishseus project.  They can be
# adjusted here or via the web UI.
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
}


def load_config() -> dict[str, object]:
    """Load the configuration from disk or return defaults."""
    if not CONFIG_FILE.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # ensure defaults for missing keys
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        merged.update({k: v for k, v in data.items() if k in merged})
        return merged
    except Exception:
        # fallback to defaults on error
        return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(config: dict[str, object]) -> None:
    """Write the configuration to disk."""
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Request Handler
# ---------------------------------------------------------------------

class FishseusRequestHandler(BaseHTTPRequestHandler):
    """
    Handle HTTP requests for the Fishseus control panel.
    """

    def _send_json(self, obj: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        try:
            content = file_path.read_bytes()
        except Exception:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        ctype = "text/html" if file_path.suffix == ".html" else "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            # Serve the UI file
            self._send_file(UI_FILE)
            return
        elif path == "/styles.css":
            # Serve the stylesheet
            self._send_file(CSS_FILE)
            return
        elif path == "/api/status":
            # Return a simple status report of the services
            status = {
                "assistant": "ready",
                "motion": "ready",
                "tts": "ready",
            }
            self._send_json(status)
            return
        elif path == "/api/config":
            self._send_json(load_config())
            return
        elif path.startswith("/api/"):
            # e.g. /api/audio -> return that subsection
            config = load_config()
            section = path[len("/api/") :]
            if section in config:
                self._send_json(config[section])
                return
            # unknown section
            self._send_json({"error": f"Unknown section '{section}'"}, status=HTTPStatus.NOT_FOUND)
            return
        else:
            # 404 for unknown GET
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        # Determine content length to read body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            data = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            data = {}
        # Route the request
        if path.startswith("/api/"):
            endpoint = path[len("/api/") :]
            # Handle talk endpoint
            if endpoint == "talk":
                message = str(data.get("message", "")).strip()
                # Use the assistant to generate a reply
                result = assistant_handle_user_text(message)
                self._send_json({"reply": result.get("reply", "")})
                return
            # Handle service reset
            if endpoint == "services/reset":
                status = reset_services()
                self._send_json({"status": status})
                return
            # Handle motor commands
            if endpoint.startswith("motor/"):
                command = endpoint[len("motor/") :]
                if command == "open_mouth":
                    # parse optional duration and speed
                    duration = data.get("duration")
                    speed = data.get("speed")
                    try:
                        duration_f = float(duration) if duration is not None else None
                    except (TypeError, ValueError):
                        duration_f = None
                    try:
                        speed_i = int(speed) if speed is not None else None
                    except (TypeError, ValueError):
                        speed_i = None
                    status = motion_open_mouth(duration=duration_f, speed=speed_i)
                    self._send_json({"status": status})
                    return
                elif command == "wiggle":
                    cycles = data.get("cycles", 1)
                    try:
                        cycles = int(cycles)
                    except (TypeError, ValueError):
                        cycles = 1
                    status = motion_wiggle(cycles)
                    self._send_json({"status": status})
                    return
                elif command == "stop":
                    # Optionally implement stop for real service if available
                    try:
                        config = load_config()
                        reset_services()
                        status = "Motion stop triggered"
                    except Exception:
                        status = "Stop not implemented"
                    self._send_json({"status": status})
                    return
                else:
                    self._send_json({"error": f"Unknown motor command '{command}'"}, status=HTTPStatus.NOT_FOUND)
                    return
            # For other endpoints, treat the endpoint name as a section of the config
            config = load_config()
            section = endpoint
            if section in config:
                if not isinstance(data, dict):
                    self._send_json({"error": "Invalid payload"}, status=HTTPStatus.BAD_REQUEST)
                    return
                # merge updated keys into config
                config_section = config.get(section, {})
                if isinstance(config_section, dict):
                    config_section.update(data)
                    config[section] = config_section
                    save_config(config)
                    # If updating motion settings, propagate to motion service stub/real service
                    if section == "motion":
                        try:
                            _motion_service.update_config(data)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    self._send_json({"status": "saved"})
                    return
                else:
                    self._send_json({"error": "Cannot update this section"}, status=HTTPStatus.BAD_REQUEST)
                    return
            else:
                self._send_json({"error": f"Unknown section '{section}'"}, status=HTTPStatus.NOT_FOUND)
                return
        # Unknown POST path
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")


def run_server(port: int) -> None:
    """Run the HTTP server on the given port."""
    # Define a threading mixin HTTP server class to handle requests in
    # separate threads.  This prevents deadlocks when long‑running
    # operations occur and allows us to call shutdown() cleanly.
    class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, FishseusRequestHandler)
    print(f"Serving Fishseus control panel on port {port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.shutdown()
    finally:
        httpd.server_close()


if __name__ == "__main__":
    # Determine port from command line, default 8000
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port '{sys.argv[1]}', using default 8000")
    # Ensure a config file exists
    if not CONFIG_FILE.exists():
        save_config(json.loads(json.dumps(DEFAULT_CONFIG)))
    run_server(port)