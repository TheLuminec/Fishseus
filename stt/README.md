# STT Module

Thin local speech-to-text service for Fishseus using
[whisper.cpp](https://github.com/ggml-org/whisper.cpp). Wraps the `whisper-cli`
executable behind a small `Service` API and transcribes WAV files either
synchronously or on a background worker thread.

**Responsibilities:** WAV → text transcription, sync + async job handling, and
wake-word detection/stripping — exposed as a `Service` for the orchestrator.

**Non-responsibilities:** no audio capture (that's `audio_service.py`), no
assistant, memory, or LLM logic.

Pipeline: `audio_service` records a WAV when speech is detected → `stt_service`
transcribes it → the orchestrator checks wake words and routes the command text
to the assistant/LLM layer.

## Configuration (`SttConfig`)

`SttConfig` is a frozen `ServiceConfig` dataclass. Defaults are anchored to the
repo root, so `SttService()` works out of the box:

| Field             | Default                                             | Purpose                                             |
| ----------------- | --------------------------------------------------- | --------------------------------------------------- |
| `module_name`     | `"stt"`                                             | Service key used by the orchestrator config.        |
| `whisper_binary`  | `<root>/stt/whisper.cpp/build/bin/whisper-cli`      | Path to the whisper.cpp CLI.                        |
| `model_path`      | `<root>/stt/whisper.cpp/models/ggml-base.en.bin`    | GGML model file.                                    |
| `language`        | `"en"`                                              | Transcription language (`-l`).                      |
| `threads`         | `2`                                                 | Whisper threads (`-t`).                             |
| `translate`       | `False`                                             | Translate to English (`-tr`).                       |
| `no_timestamps`   | `True`                                              | Suppress timestamps (`-nt`).                        |
| `print_special`   | `False`                                             | Print special tokens (`-ps`).                       |
| `print_progress`  | `False`                                             | Print progress; `False` adds `-np`.                 |
| `timeout_s`       | `30.0`                                              | Per-call whisper.cpp timeout.                       |
| `wake_words`      | `("fish",)`                                         | Terms checked against the transcript.               |
| `strip_wake_word` | `False`                                             | Remove the matched wake word from `command_text`.   |
| `extra_args`      | `()`                                                | Extra raw args appended to the whisper.cpp command. |

`config.validate()` raises `SttServiceError` if the binary or model file is
missing. It runs automatically inside `initialize()`.

## Lifecycle API

`SttService` implements the standard `Service` interface:

- `initialize()` – validates config and starts the background worker thread.
  Call before transcribing.
- `shutdown()` – signals the worker to stop and joins it.
- `reset()` – `shutdown()` then `initialize()`.
- `status()` – returns `{enabled, service, worker_thread_alive, job_queue_size, result_queue_size}`.

## Transcription API

- `transcribe_file(wav_path)` – **blocking** transcription; returns a
  `TranscriptionResult`.
- `transcribe_file_async(wav_path, callback=None)` – queues a job and returns
  immediately. With a `callback`, the result is delivered to it; without one,
  the result is placed on the queue for `get_result()`.
- `get_result(timeout=None)` – returns the next queued async result, or `None`
  if the timeout elapses.
- `contains_wake_word(text) -> (bool, wake_word | None)` – word-boundary
  wake-word match (so `fish` won't match `selfish`).

### `TranscriptionResult`

| Field           | Meaning                                              |
| --------------- | ---------------------------------------------------- |
| `text`          | Full cleaned transcript.                             |
| `command_text`  | Transcript with the wake word stripped (if enabled). |
| `wake_detected` | Whether a wake word matched.                         |
| `wake_word`     | Which wake word matched, or `None`.                  |
| `elapsed_s`     | Transcription wall-clock time.                       |
| `source_path`   | The WAV that was transcribed.                        |
| `raw_stdout` / `raw_stderr` | Unprocessed whisper.cpp output.          |

## How it works

- `initialize()` validates the config and starts a daemon worker thread that
  drains an internal job queue.
- Each transcription shells out to `whisper-cli` with flags built from
  `SttConfig`, then parses the transcript out of stdout (stripping stray
  timestamp/log lines and Whisper special tokens like `<|endoftext|>`).
- Wake words are matched on a normalized copy of the text; when
  `strip_wake_word` is set, the first occurrence is removed from `command_text`.
- Async jobs run on the worker: a **failed transcription is logged and skipped**
  so one bad recording never kills the worker, and a raising callback is caught
  the same way. Results with a callback go to the callback; otherwise they go to
  the result queue.

> **Note:** when only callbacks are used, results are *not* placed on the result
> queue, so there's nothing to drain. If you use `get_result()`, drain it —
> results accumulate until you do.

## Usage

### Blocking

```python
stt = SttService()
stt.initialize()
result = stt.transcribe_file("tmp/latest_speech.wav")
if result.wake_detected:
    print(result.command_text)
stt.shutdown()
```

### Async (orchestrator)

```python
stt = SttService(SttConfig(**config.get("stt", {})))
stt.initialize()

# Deliver results via callback...
stt.transcribe_file_async(wav_path, lambda r: handle(r) if r.wake_detected else None)

# ...or poll for them.
stt.transcribe_file_async(wav_path)
result = stt.get_result(timeout=5.0)
if result and result.wake_detected:
    handle(result.command_text)

stt.shutdown()
```

### Inspecting status

```python
stt = SttService()
stt.initialize()
print(stt.status())
# {'enabled': True, 'service': 'ok', 'worker_thread_alive': True,
#  'job_queue_size': 0, 'result_queue_size': 0}
stt.shutdown()
```

## Requirements

- A built `whisper.cpp` with `whisper-cli` at `whisper_binary` (default:
  `stt/whisper.cpp/build/bin/whisper-cli`).
- A GGML model at `model_path` — e.g. `ggml-tiny.en.bin`, `ggml-base.en.bin`, or
  `ggml-small.en.bin` under `stt/whisper.cpp/models/`.
