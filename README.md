# Fishseus

Fishseus is a modular voice-assistant stack for a motorized Billy Bass-style fish.

## Module documentation
- [`audio/README.md`](audio/README.md): microphone capture and PCM queueing APIs.
- [`stt/README.md`](stt/README.md): whisper.cpp speech-to-text wrapper APIs.
- [`tts/README.md`](tts/README.md): Piper text-to-speech generation and playback APIs.
- [`llm/README.md`](llm/README.md): OpenAI-compatible chat completion client APIs.
- [`assistant/README.md`](assistant/README.md): assistant orchestration, memory, and tool APIs.
- [`motion/README.md`](motion/README.md): I2C motor control (ATtiny1614 HAT) and animation APIs.
- [`orchestrator/README.md`](orchestrator/README.md): end-to-end demo workflows.
- [`config/README.md`](config/README.md): prompt/config assets and usage.
- [`bluetooth/README.md`](bluetooth/README.md): Bluetooth speaker pairing and connection (BlueZ).
- [`spotify/README.md`](spotify/README.md): voice-requested Spotify playback via spotipy.
- [`access_point/README.md`](access_point/README.md): the isolated "Fishseus" Wi-Fi network for the turret ESP32s.

## End-to-end flow
1. `audio` captures speech and writes/streams WAV PCM.
2. `stt` transcribes speech and extracts wake-word command text.
3. `assistant` + `llm` generate a response and optional tool calls.
4. `motion` executes fish animations.
5. `tts` (optional orchestrator path) vocalizes assistant output.
