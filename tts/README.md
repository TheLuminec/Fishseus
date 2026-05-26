# TTS Module

Provides local speech generation with Piper and optional WAV playback through ALSA (`aplay`).

## Main API (`tts_service.py`)
- `TtsService.available_voices()` – lists `.onnx` voices in `voices_dir`.
- `TtsService.set_voice(voice_name)` – switches active voice.
- `TtsService.synthesize(text, output_path=None, voice=None)` – creates WAV from text.
- `TtsService.play_wav(wav_path, blocking=True)` – plays WAV to configured audio device.
- `TtsService.speak(text, voice=None, blocking=True)` – synthesize + play convenience method.

## How it works
- Validates voice model/config files.
- Runs Piper via subprocess, sending text on stdin.
- Writes speech WAV to output directory.
- Uses `aplay` for playback when requested.

## Typical usage
```python
tts = TtsService()
wav = tts.synthesize("Hello from Fishseus")
tts.play_wav(wav)
```
