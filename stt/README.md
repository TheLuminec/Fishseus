# STT Module

Wraps local `whisper.cpp` transcription behind a simple synchronous + asynchronous Python API.

## Main API (`stt_service.py`)
- `SttService.initialize()` – validates binary/model paths and starts worker thread.
- `SttService.transcribe_file(wav_path)` – blocking WAV transcription.
- `SttService.transcribe_file_async(wav_path, callback=None)` – queues background transcription.
- `SttService.get_result(timeout=None)` – retrieves async results.
- `SttService.contains_wake_word(text)` – wake-word detection helper.
- `SttService.shutdown()` – stops worker thread.

## How it works
- Calls `whisper-cli` with `SttConfig` settings.
- Extracts transcript text from CLI output.
- Detects/optionally strips configured wake words.
- Returns `TranscriptionResult` with transcript, command text, timing, and raw stdout/stderr.

## Typical usage
```python
stt.initialize()
result = stt.transcribe_file("tmp/latest_speech.wav")
if result.wake_detected:
    print(result.command_text)
```
