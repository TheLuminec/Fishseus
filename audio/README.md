# Audio Module

Provides low-overhead microphone capture through `arecord` and streams PCM chunks to the rest of the stack.

## Main API (`audio_service.py`)
- `AudioService.initialize()` – prepares service state.
- `AudioService.start_capture(amplitude_callback=None)` – starts background capture thread.
- `AudioService.get_chunk(timeout=None)` – returns next `PcmChunk` (`data`, `timestamp`, `amplitude`).
- `AudioService.record_fixed_wav(output_path, duration_s)` – records a fixed-duration WAV.
- `AudioService.record_until_silence(output_path)` – waits for speech, then stops after silence.
- `AudioService.stop_capture()` / `shutdown()` – stops capture and cleans up `arecord` process.

## How it works
- Spawns `arecord` with configured sample rate/channels.
- Reads raw PCM in small chunks on a daemon thread.
- Computes normalized amplitude and pushes chunks into a bounded queue.
- Optional amplitude callback is emitted at configured intervals.

## Typical usage
```python
from audio_service import AudioService, AudioConfig

audio = AudioService(AudioConfig(device="default"))
audio.initialize()
audio.start_capture()
chunk = audio.get_chunk(timeout=0.1)
audio.shutdown()
```
