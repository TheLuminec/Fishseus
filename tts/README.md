# TTS Module

Thin local text-to-speech service for Fishseus using [Piper](https://github.com/rhasspy/piper).
Generates WAV speech locally and plays it through ALSA (`aplay`), keeping Piper
resident as a persistent daemon so the ONNX voice model stays warm in memory
between utterances.

**Responsibilities:** local WAV synthesis, persistent daemon management, voice
switching, and ALSA playback — exposed as a small `Service` API for the
orchestrator.

**Non-responsibilities:** no fish motion, assistant, memory, or LLM logic.

## Configuration (`TtsConfig`)

`TtsConfig` is a frozen `ServiceConfig` dataclass. All fields have defaults
anchored to the repo root, so `TtsService()` works out of the box:

| Field           | Default                                | Purpose                                        |
| --------------- | -------------------------------------- | ---------------------------------------------- |
| `module_name`   | `"tts"`                                | Service key used by the orchestrator config.   |
| `piper_binary`  | `<root>/tts/.venv/bin/piper`           | Absolute path to the Piper executable.         |
| `voices_dir`    | `<root>/tts/voices`                    | Folder of `*.onnx` voices + `*.onnx.json`.     |
| `default_voice` | `"en_US-arctic-medium"`                | Voice filename stem (no extension).            |
| `audio_device`  | `"plughw:0,0"`                         | ALSA playback target for `aplay`.              |
| `output_dir`    | `<root>/tmp/tts`                       | Where generated WAVs are written.              |
| `timeout_s`     | `30.0`                                 | Synthesis timeout (daemon and subprocess).     |
| `persistent`    | `True`                                 | Keep Piper resident; `False` = one proc/call.  |

`config.validate()` raises `TtsServiceError` if the binary, voices directory, or
default voice model is missing. It runs automatically inside `initialize()`.

## Lifecycle API

`TtsService` implements the standard `Service` interface:

- `initialize()` – validates config, ensures `output_dir` exists, and starts the
  persistent Piper daemon (when `persistent=True`). Call before synthesizing.
- `shutdown()` – terminates the daemon cleanly.
- `reset()` – restarts the daemon with the current voice (use if it wedges).
- `status()` – returns `{enabled, service, daemon_alive, current_voice}`.

## Speech API

- `available_voices()` – lists `.onnx` voice stems found in `voices_dir`.
- `set_voice(voice_name)` – switches the active voice for all future speech and
  restarts the daemon so the new model is pre-loaded.
- `synthesize(text, output_path=None, voice=None)` – renders WAV from text and
  returns its `Path`. Uses the resident daemon when the requested voice matches
  the loaded one, otherwise falls back to a one-shot subprocess.
- `play_wav(wav_path, blocking=True)` – plays a WAV to the configured ALSA device.
- `speak(text, voice=None, blocking=True)` – synthesize + play convenience method.

## How it works

- On `initialize()`, Piper is launched once in `--json-input` mode with the
  default voice loaded, and kept alive as a daemon.
- Each `synthesize()` call sends a `{"text", "output_file"}` JSON line on the
  daemon's stdin; completion is detected by polling for the output WAV.
- If the daemon isn't running (or `persistent=False`, or a different voice is
  requested), synthesis falls back to spawning a fresh Piper subprocess.
- Daemon start/stop is guarded by a reentrant lock so lifecycle transitions and
  in-flight synthesis don't collide.

## Usage

### Basic

```python
tts = TtsService()
tts.initialize()                       # starts the persistent piper daemon
wav = tts.synthesize("Behold, I awaken.")
tts.play_wav(wav)
tts.shutdown()
```

### Orchestrator

```python
tts = TtsService(TtsConfig(**config.get("tts", {})))
tts.initialize()
tts.speak("Hello, world!")                       # default voice, synthesize + play
tts.speak("Hello, world!", voice="some_voice")   # override voice for this line only
tts.set_voice("some_voice")                      # switch voice for all future speech
tts.speak("Hello, world!")                       # uses the new default voice
tts.reset()                                      # restart the daemon (use if broken)
tts.shutdown()                                   # clean up the daemon
```

### Inspecting voices and status

```python
tts = TtsService()
print(tts.available_voices())          # ['en_US-arctic-medium', ...]

tts.initialize()
print(tts.status())
# {'enabled': True, 'service': 'ok', 'daemon_alive': True,
#  'current_voice': 'en_US-arctic-medium'}
tts.shutdown()
```

### One-shot mode (no resident daemon)

```python
tts = TtsService(TtsConfig(persistent=False))
tts.initialize()
tts.speak("Spawns a fresh piper process per utterance.")
tts.shutdown()
```

## Requirements

- Piper installed at `piper_binary` (default: the `tts/.venv`).
- Voice models in `voices_dir` — each voice needs both `<name>.onnx` and
  `<name>.onnx.json`.
- `aplay` (from `alsa-utils`) for playback.
