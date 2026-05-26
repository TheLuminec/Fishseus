# Orchestrator Module

Contains end-to-end demo loops that connect audio, STT, assistant/LLM, motion, and optional TTS.

## Entry points
- `orchestrator_demo.py` – voice loop without TTS playback.
- `orchestrator_demo_tts.py` – voice loop with Piper TTS + synchronized fish mouth animation.

## How the API is used
- Initializes each service with module-specific config.
- Captures speech from microphone (`AudioService`).
- Transcribes WAV (`SttService`) and checks wake words.
- Routes command text to `AssistantService`/`LlmService`.
- Runs tool actions (e.g., fish motion), and in TTS mode synthesizes/plays spoken replies.

## Run examples
```bash
python orchestrator/orchestrator_demo.py
python orchestrator/orchestrator_demo_tts.py
```
