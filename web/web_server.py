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
from pathlib import Path


# ---------------------------------------------------------------------
# Configuration and stubs
# ---------------------------------------------------------------------

# Location of the UI HTML file and the config file.  We use the
# directory of this script as the base.
ROOT_DIR = Path(__file__).resolve().parent
UI_FILE = ROOT_DIR / "fishseus_ui.html"
CONFIG_FILE = ROOT_DIR / "fish_config.json"

# Import stubs for assistant and motion.  These provide simple
# stand‑ins for the real services.  If the real modules are available
# in your environment, you can swap these imports accordingly.
try:
    from assistant.assistant_service import handle_user_text as assistant_handle_user_text
except ImportError:
    # Fallback to echo if stub cannot be imported
    def assistant_handle_user_text(text: str):
        return {"reply": f"Echo: {text}"}

try:
    from motion.motion_service import MotionServiceStub, MotionConfig
    # Create a single motion service instance.  In a real system this
    # would be a long‑running service controlling GPIO.
    _motion_service = MotionServiceStub()
except ImportError:
    # Define a minimal stub if import fails
    class _SimpleMotion:
        def open_mouth(self):
            return "Mouth opening (fallback stub)"
        def wiggle(self, cycles=1):
            return f"Wiggle {cycles} time(s) (fallback stub)"
        def update_config(self, data):
            pass
        def get_config(self):
            return {}
    _motion_service = _SimpleMotion()


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
                # Use the assistant stub to generate a reply
                result = assistant_handle_user_text(message)
                self._send_json({"reply": result.get("reply", "")})
                return
            # Handle motor commands
            elif endpoint.startswith("motor/"):
                command = endpoint[len("motor/") :]
                if command == "open_mouth":
                    status = _motion_service.open_mouth()
                    self._send_json({"status": status})
                    return
                elif command == "wiggle":
                    cycles = int(data.get("cycles", 1))
                    status = _motion_service.wiggle(cycles)
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
                    # If updating motion settings, propagate to motion service stub
                    if section == "motion":
                        # update internal stub config so the test buttons reflect new values
                        # Only update nested dicts
                        if "motors" in data or any(k in data for k in ["pwm_frequency", "body_wiggle_time", "tail_wiggle_time", "mouth_open_time", "mouth_close_time", "envelope_window_s"]):
                            try:
                                _motion_service.update_config(data)
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
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")


def run_server(port: int) -> None:
    """Run the HTTP server on the given port."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, FishseusRequestHandler)
    print(f"Serving Fishseus control panel on port {port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
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