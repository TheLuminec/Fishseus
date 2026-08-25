# Web Server / Control Panel

The `web` module provides a Flask-based web server and a modernized, responsive, dark ocean-themed control panel UI to configure and interact with Fishseus.

## Features

- **Graceful Degradation:** The server attempts to import hardware-dependent services (like `motion_service`). If they fail (e.g., when testing on a non-Raspberry Pi machine), the UI cleanly disables their respective controls and shows them as "Unavailable" in the status dashboard.
- **Unified Dashboard:** Real-time status indicators for LLM, Assistant, Motion, and TTS.
- **Chat Interface:** A built-in chat UI to interact with Fishseus directly from the browser, bypassing the microphone.
- **Configuration Management:** Save and load configurations for audio, STT, LLM, TTS, and motion directly to `fish_config.json`.
- **Motor Testing:** Dedicated cards for each motor to test speeds, durations, and specific animations (e.g., wiggle, speak placeholder).
- **Personality Editor:** Live-edit the system prompt that defines the assistant's character.

## Prerequisites

The web server needs `flask` (and `PyJWT[crypto]` for Cloudflare Access). On the
Pi, install them into a virtualenv created with `--system-site-packages` so the
hardware services (`RPi.GPIO`, `gpiozero`, `requests`) remain importable:

```bash
python -m venv web/.venv --system-site-packages
web/.venv/bin/pip install -r web/requirements.txt
```

## Usage

Run with the venv's interpreter (the orchestrator imports the web server
in-process, so it needs the same interpreter):

```bash
web/.venv/bin/python web/web_server.py [port]     # standalone panel
web/.venv/bin/python fishseus.py --port 8000      # full assistant + web UI
```

By default the server now binds to **`127.0.0.1:8000`** (loopback only), intended
to sit behind a Cloudflare Tunnel — see [DEPLOYMENT.md](DEPLOYMENT.md). For plain
LAN access, set `FISHSEUS_BIND_HOST=0.0.0.0` (or `web.bind_host` in the config)
and open `http://<your-pi-ip>:8000/`.

## API Endpoints

The Flask app exposes JSON endpoints for the frontend:

- `GET /api/status`: Returns current availability of services.
- `GET /api/config`: Returns the merged configuration dictionary.
- `POST /api/<section>`: Updates a specific configuration section (e.g., `audio`, `motion`).
- `GET/POST /api/personality`: Reads or writes the raw `personality_prompt.txt` file.
- `POST /api/talk`: Sends a text message to the assistant and returns the generated speech and motion tags.
- `POST /api/motor/<command>`: Executes test motions (`open_mouth`, `wiggle`, `speak_placeholder`, `stop`).
- `POST /api/services/reset`: Tears down and reinitializes all backend service instances.

## UI Theming

The UI (`fishseus_ui.html` and `styles.css`) uses a custom "dark ocean" glassmorphism theme:
- Typography: Inter and JetBrains Mono (loaded via Google Fonts).
- Colors: Deep navy backgrounds, bioluminescent cyan (`#00b4d8`) accents, and electric teal highlights.
- Components: Animated toast notifications, chat bubbles with typing indicators, and responsive sidebars.
