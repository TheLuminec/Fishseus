# Fishseus

Fishseus is a modular voice-assistant stack for a motorized Billy Bass-style fish.

## Module documentation
- [`audio/README.md`](audio/README.md): microphone capture and PCM queueing APIs.
- [`stt/README.md`](stt/README.md): whisper.cpp speech-to-text wrapper APIs.
- [`tts/README.md`](tts/README.md): Piper text-to-speech generation and playback APIs.
- [`llm/README.md`](llm/README.md): OpenAI-compatible chat completion client APIs.
- [`assistant/README.md`](assistant/README.md): assistant orchestration, memory, and tool APIs.
- [`motion/README.md`](motion/README.md): GPIO motor control and animation APIs.
- [`orchestrator/README.md`](orchestrator/README.md): end-to-end demo workflows.
- [`config/README.md`](config/README.md): prompt/config assets and usage.

## End-to-end flow
1. `audio` captures speech and writes/streams WAV PCM.
2. `stt` transcribes speech and extracts wake-word command text.
3. `assistant` + `llm` generate a response and optional tool calls.
4. `motion` executes fish animations.
5. `tts` (optional orchestrator path) vocalizes assistant output.
