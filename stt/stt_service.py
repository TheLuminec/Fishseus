"""
stt_service.py

Speech-to-text service for the Fishseus/Billy Bass assistant project.

This service wraps a local whisper.cpp executable and exposes a clean Python API
for the orchestrator. It intentionally does not capture audio itself; audio
capture remains the responsibility of audio_service.py.

Initial integration mode:
- audio_service records a WAV file when speech is detected
- stt_service transcribes that WAV with whisper.cpp
- orchestrator checks for wake words and routes text to the assistant/LLM layer

Why file-based first?
- Easier to debug
- Works well with whisper.cpp CLI
- Avoids fighting live-streaming edge cases early
- Keeps STT isolated and replaceable later

Later upgrades:
- streaming whisper.cpp mode
- faster wake-word prefilter
- remote STT backend
- multiple STT model profiles
"""

from __future__ import annotations

import sys
import json
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Optional


@dataclass(frozen=True)
class SttConfig:
    """
    Configuration for whisper.cpp CLI transcription.

    Common model examples:
        models/ggml-tiny.en.bin
        models/ggml-base.en.bin
        models/ggml-small.en.bin
    """

    whisper_binary: str = "./whisper.cpp/build/bin/whisper-cli"
    model_path: str = "./whisper.cpp/models/ggml-base.en.bin"

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
    strip_wake_word: bool = True

    # Optional extra whisper.cpp args for experimentation.
    extra_args: tuple[str, ...] = field(default_factory=tuple)


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


class SttService:
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
        with self._lock:
            if self._initialized:
                return

            binary = Path(self.config.whisper_binary)
            model = Path(self.config.model_path)

            if not binary.exists():
                raise FileNotFoundError(f"whisper.cpp binary not found: {binary}")

            if not model.exists():
                raise FileNotFoundError(f"whisper.cpp model not found: {model}")

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
            if not self._initialized:
                return

            self._stop_event.set()
            self._job_queue.put((Path("__shutdown__"), None))

            if self._worker_thread is not None:
                self._worker_thread.join(timeout=2.0)
                self._worker_thread = None

            self._initialized = False

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
            raise RuntimeError("SttService.initialize() must be called first")

        source_path = Path(wav_path)
        if not source_path.exists():
            raise FileNotFoundError(f"WAV file not found: {source_path}")

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
            raise RuntimeError("SttService.initialize() must be called first")

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

            if str(wav_path) == "__shutdown__":
                break

            try:
                result = self.transcribe_file(wav_path)
                self._result_queue.put(result)
                if callback is not None:
                    callback(result)
            except Exception as exc:
                print(f"[SttService] transcription failed for {wav_path}: {exc}")
            finally:
                self._job_queue.task_done()

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
            raise RuntimeError(
                "whisper.cpp failed with code "
                f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        return completed.stdout, completed.stderr

    def _build_command(self, wav_path: Path) -> list[str]:
        cfg = self.config

        cmd = [
            cfg.whisper_binary,
            "-m", cfg.model_path,
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


# ----------------------------------------------------------------------
# Manual test runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Edit these paths to match your whisper.cpp install.
    config = SttConfig(
        whisper_binary="./whisper.cpp/build/bin/whisper-cli",
        model_path="./whisper.cpp/models/ggml-base.en.bin",
        threads=4,
        wake_words=("fish", "hey fish"),
    )

    if len(sys.argv) < 2:
        print("Usage: python3 stt_service.py path/to/audio.wav")
        raise SystemExit(1)

    stt = SttService(config)
    stt.initialize()

    try:
        result = stt.transcribe_file(sys.argv[1])
        print("--- Transcription Result ---")
        print(f"Text:          {result.text}")
        print(f"Wake detected: {result.wake_detected}")
        print(f"Wake word:     {result.wake_word}")
        print(f"Command text:  {result.command_text}")
        print(f"Elapsed:       {result.elapsed_s:.2f}s")
    finally:
        stt.shutdown()
