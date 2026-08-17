"""
stt_service.py

Thin local speech-to-text service for Fishseus using whisper.cpp.

Responsibilities:
- Transcribe a WAV file to text with the whisper.cpp CLI
- Run transcription synchronously, or on a background worker thread
- Detect (and optionally strip) configured wake words
- Keep a simple API for the orchestrator

Non-responsibilities:
- No audio capture (that is audio_service.py)
- No assistant logic
- No memory
- No LLM logic

Pipeline: audio_service records a WAV on speech -> stt_service transcribes it ->
the orchestrator checks wake words and routes command text to the assistant/LLM.

Example (blocking):
    stt = SttService()
    stt.initialize()                       # validates config, starts the worker
    result = stt.transcribe_file("tmp/latest_speech.wav")
    if result.wake_detected:
        print(result.command_text)
    stt.shutdown()

Orchestrator usage (async):
    stt = SttService(SttConfig(**config.get("stt", {})))
    stt.initialize()
    stt.transcribe_file_async(wav_path)             # queue, returns immediately
    stt.transcribe_file_async(wav_path, on_result)  # or deliver via callback
    result = stt.get_result(timeout=5.0)            # poll for the next result
    if result and result.wake_detected:
        handle(result.command_text)
    stt.shutdown()                                  # stops the worker
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Optional

from services import Service, ServiceConfig, ServiceError, ROOT_DIR

class SttServiceError(ServiceError):
    pass

@dataclass(frozen=True)
class SttConfig(ServiceConfig):
    """
    Configuration for whisper.cpp CLI transcription.

    Common model examples:
        models/ggml-tiny.en.bin
        models/ggml-base.en.bin
        models/ggml-small.en.bin
    """
    module_name: str = "stt"

    whisper_binary: Path = ROOT_DIR / "stt" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    model_path: Path = ROOT_DIR / "stt" / "whisper.cpp" / "models" / "ggml-base.en.bin"

    language: str = "en"
    threads: int = 2

    # Good defaults for assistant commands.
    translate: bool = False
    no_timestamps: bool = True
    print_special: bool = False
    print_progress: bool = False

    # Limit runaway calls. Keep this comfortably above expected utterance length.
    timeout_s: float = 30.0

    # Wake terms are checked after transcription.
    wake_words: tuple[str, ...] = ("fish",)

    # If true, remove common wake words from the returned command text.
    strip_wake_word: bool = False

    # Optional extra whisper.cpp args for experimentation.
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> bool:
        if not self.whisper_binary.exists():
            raise SttServiceError(f"whisper.cpp binary not found: {self.whisper_binary}")

        if not self.model_path.exists():
            raise SttServiceError(f"whisper.cpp model not found: {self.model_path}")

        return True


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    command_text: str
    wake_detected: bool
    wake_word: Optional[str]
    elapsed_s: float
    source_path: Path
    raw_stdout: str = ""
    raw_stderr: str = ""


class SttService(Service):
    """
    Thin, replaceable STT layer around whisper.cpp.

    Public API:
        initialize()
        transcribe_file(path)
        transcribe_file_async(path, callback=None)
        get_result(timeout=None)
        contains_wake_word(text)
        shutdown()

    This service uses a small worker queue for async transcription so the
    orchestrator can submit speech recordings without blocking the main loop.
    """

    def __init__(self, config: SttConfig = SttConfig()) -> None:
        self.config = config
        self._initialized = False
        self._stop_event = threading.Event()
        self._job_queue: Queue[tuple[Path, Optional[Callable[[TranscriptionResult], None]]]] = Queue()
        self._result_queue: Queue[TranscriptionResult] = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.config.validate()

        with self._lock:
            if self._initialized:
                return

            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="SttWorkerThread",
                daemon=True,
            )
            self._worker_thread.start()
            self._initialized = True

    def shutdown(self) -> None:
        with self._lock:
            self._stop_event.set()

            if self._worker_thread is not None:
                # The worker only checks _stop_event between jobs; if it's mid
                # transcription (whisper.cpp can run up to timeout_s) the join
                # may time out.  Keep the reference until it actually exits so
                # status() doesn't falsely report the worker as gone.
                self._worker_thread.join(timeout=2.0)
                if not self._worker_thread.is_alive():
                    self._worker_thread = None

            self._initialized = False

    def reset(self) -> bool:
        self.shutdown()
        self.initialize()

        return True

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "service": "ok" if self._initialized else "uninitialized",
            "worker_thread_alive": self._worker_thread.is_alive() if self._worker_thread else False,
            "job_queue_size": self._job_queue.qsize(),
            "result_queue_size": self._result_queue.qsize()
        }

    # ------------------------------------------------------------------
    # Public synchronous API
    # ------------------------------------------------------------------
    def transcribe_file(self, wav_path: str | Path) -> TranscriptionResult:
        """
        Blocking transcription call.

        This is useful for tests or simple orchestrator flows:
            wav = audio.record_until_silence(...)
            result = stt.transcribe_file(wav)
        """
        if not self._initialized:
            raise SttServiceError("SttService.initialize() must be called first")

        source_path = Path(wav_path)
        if not source_path.exists():
            raise SttServiceError(f"WAV file not found: {source_path}")

        start = time.monotonic()
        stdout, stderr = self._run_whisper_cli(source_path)
        elapsed = time.monotonic() - start

        text = self._extract_text(stdout)
        wake_detected, wake_word = self.contains_wake_word(text)
        command_text = self._make_command_text(text, wake_word)

        return TranscriptionResult(
            text=text,
            command_text=command_text,
            wake_detected=wake_detected,
            wake_word=wake_word,
            elapsed_s=elapsed,
            source_path=source_path,
            raw_stdout=stdout,
            raw_stderr=stderr,
        )

    # ------------------------------------------------------------------
    # Public asynchronous API
    # ------------------------------------------------------------------
    def transcribe_file_async(
        self,
        wav_path: str | Path,
        callback: Optional[Callable[[TranscriptionResult], None]] = None,
    ) -> None:
        """
        Queue a file for transcription and return immediately.

        Results can be received either through callback or get_result().
        """
        if not self._initialized:
            raise SttServiceError("SttService.initialize() must be called first")

        self._job_queue.put((Path(wav_path), callback))

    def get_result(self, timeout: Optional[float] = None) -> Optional[TranscriptionResult]:
        """
        Return next async transcription result, or None if no result arrives.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except Empty:
            return None

    # ------------------------------------------------------------------
    # Wake-word helpers
    # ------------------------------------------------------------------
    def contains_wake_word(self, text: str) -> tuple[bool, Optional[str]]:
        normalized = self._normalize_text(text)

        for wake_word in self.config.wake_words:
            normalized_wake = self._normalize_text(wake_word)
            # Word-boundary match so "fish" does not match "selfish".
            pattern = rf"\b{re.escape(normalized_wake)}\b"
            if re.search(pattern, normalized):
                return True, wake_word

        return False, None

    def _make_command_text(self, text: str, wake_word: Optional[str]) -> str:
        cleaned = text.strip()

        if not self.config.strip_wake_word or not wake_word:
            return cleaned

        # Remove the first wake-word occurrence in a forgiving way.
        pattern = rf"\b{re.escape(wake_word)}\b[,.!?;:\- ]*"
        cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
        return cleaned

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                wav_path, callback = self._job_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                result = self.transcribe_file(wav_path)
            except Exception as exc:
                print(f"[SttService] transcription failed for {wav_path}: {exc}")
                continue                        # keep the worker alive
            finally:
                self._job_queue.task_done()
            
            if callback is None:
                self._result_queue.put(result)
            else:
                try:
                    callback(result)
                except Exception as exc:
                    print(f"[SttService] result callback failed: {exc}")

    # ------------------------------------------------------------------
    # whisper.cpp integration
    # ------------------------------------------------------------------
    def _run_whisper_cli(self, wav_path: Path) -> tuple[str, str]:
        cmd = self._build_command(wav_path)

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_s,
        )

        if completed.returncode != 0:
            raise SttServiceError(
                "whisper.cpp failed with code "
                f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        return completed.stdout, completed.stderr

    def _build_command(self, wav_path: Path) -> list[str]:
        cfg = self.config

        cmd = [
            str(cfg.whisper_binary),
            "-m", str(cfg.model_path),
            "-f", str(wav_path),
            "-l", cfg.language,
            "-t", str(cfg.threads),
        ]

        if cfg.translate:
            cmd.append("-tr")

        if cfg.no_timestamps:
            cmd.append("-nt")

        if cfg.print_special:
            cmd.append("-ps")

        if not cfg.print_progress:
            cmd.append("-np")

        cmd.extend(cfg.extra_args)
        return cmd

    def _extract_text(self, stdout: str) -> str:
        """
        Extract text from whisper.cpp stdout.

        With -nt -np, whisper.cpp usually prints fairly clean text, but builds
        differ a bit. This strips obvious timestamp/log-ish lines while keeping
        the actual transcript.
        """
        lines: list[str] = []

        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Drop timestamp prefix if one appears anyway.
            line = re.sub(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*", "", line)

            # Skip common whisper.cpp diagnostic lines if they leak to stdout.
            lowered = line.lower()
            if lowered.startswith("whisper_") or lowered.startswith("system_info"):
                continue

            lines.append(line)

        text = " ".join(lines)
        # Strip Whisper special tokens (<|endoftext|>, <|notimestamps|>, etc.)
        text = re.sub(r"<\|[^|]*\|>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9' ]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

