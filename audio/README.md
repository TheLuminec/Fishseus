# Audio Module

Thin local audio-capture service for Fishseus using ALSA/`arecord`. Captures PCM
from a USB microphone on a background thread, exposes live amplitude, and records
speech to WAV — cheaply, so STT can run in parallel.

**Responsibilities:** microphone capture, live amplitude, fixed/until-silence WAV
recording, bounded PCM streaming to an STT worker.

**Non-responsibilities:** no Whisper/STT, TTS, LLM, or motion logic.

## Configuration (`AudioConfig`)

| Field                  | Default     | Purpose                                             |
| ---------------------- | ----------- | --------------------------------------------------- |
| `module_name`          | `"audio"`   | Service key in the orchestrator config.             |
| `device`               | `"default"` | ALSA capture device (`arecord -L` to list).         |
| `sample_rate`          | `16000`     | Capture rate (Hz).                                  |
| `channels`             | `1`         | Channel count.                                      |
| `sample_width_bytes`   | `2`         | Bytes/sample (S16_LE).                              |
| `chunk_frames`         | `1024`      | Frames per read (~64 ms at 16 kHz).                 |
| `max_queue_chunks`     | `64`        | Bounded queue cap; oldest dropped when full.        |
| `amplitude_interval_s` | `0.5`       | How often the amplitude callback fires.             |
| `speech_threshold`     | `0.0025`    | Amplitude that starts a speech recording.           |
| `silence_threshold`    | `0.0015`    | Below this counts as silence.                       |
| `silence_timeout_s`    | `1.0`       | Silence duration that ends a recording.             |
| `max_record_seconds`   | `12.0`      | Hard cap on wait/record length.                     |
| `pre_roll_seconds`     | `0.4`       | Audio kept from just before speech starts.          |

`config.validate()` raises `AudioServiceError` on a non-positive sample rate,
`channels < 1`, or an unsupported sample width. It runs in `initialize()`.

## Lifecycle API

- `initialize()` – validate config and mark ready (idempotent).
- `start_capture(amplitude_callback=None)` – spawn `arecord` + the capture thread.
- `stop_capture()` – stop the thread and terminate `arecord`.
- `shutdown()` – `stop_capture()` and mark uninitialized.
- `reset()` – `shutdown()` then `initialize()`.
- `status()` – `{enabled, service, capturing, queue_size, latest_amplitude}`.

## Capture API

- `get_chunk(timeout=None)` – next `PcmChunk` (`data`, `timestamp`, `amplitude`), or `None`.
- `latest_amplitude()` – most recent normalized amplitude (~0.0–1.0).
- `record_fixed_wav(output_path, duration_s)` – record a fixed-length WAV.
- `record_until_silence(output_path)` – wait for speech, record, stop after silence; returns the path or `None`.

## How it works

- Spawns `arecord` at the configured rate/channels and reads raw PCM in small
  chunks on a daemon thread.
- Each chunk gets a normalized mean-absolute amplitude and is pushed to a bounded
  queue (oldest dropped if STT falls behind, so capture stays real-time).
- The optional amplitude callback fires at `amplitude_interval_s` for the web UI.

## Usage

```python
audio = AudioService(AudioConfig(device="plughw:1,0"))
audio.initialize()
audio.start_capture(amplitude_callback=print)
wav = audio.record_until_silence("tmp/latest_speech.wav")
audio.shutdown()
```

## Requirements

- `arecord` (from `alsa-utils`) and a working capture device.
