"""
audio_service.py

Thin local audio-capture service for Fishseus using ALSA/arecord.

Responsibilities:
- Capture PCM from a USB microphone through arecord (low overhead)
- Expose live amplitude for the web UI graph
- Record a fixed-duration WAV, or record speech until silence
- Stream bounded PCM chunks to an STT worker without unbounded RAM growth

Non-responsibilities:
- No Whisper/STT, no TTS, no LLM, no motion logic — those are separate services
  coordinated by the orchestrator

Example:
    audio = AudioService(AudioConfig(device="plughw:1,0"))
    audio.initialize()
    audio.start_capture(amplitude_callback=print)
    wav = audio.record_until_silence("tmp/latest_speech.wav")
    audio.shutdown()

Orchestrator usage:
    audio = AudioService(AudioConfig(**config.get("audio", {})))
    audio.initialize()
    audio.start_capture(amplitude_callback=on_amplitude)
    ...
    audio.stop_capture()   # while the fish is speaking
    audio.start_capture(amplitude_callback=on_amplitude)
    audio.shutdown()
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Callable, Optional

from services import Service, ServiceConfig, ServiceError


class AudioServiceError(ServiceError):
    pass


@dataclass(frozen=True)
class AudioConfig(ServiceConfig):
    module_name: str = "audio"

    # Use `arecord -L` to find a better device string if needed.
    # For a simple USB mic, "default" or "plughw:<card>,<device>" often works.
    device: str = "default"

    sample_rate: int = 16000
    channels: int = 1
    sample_width_bytes: int = 2  # S16_LE

    # Small chunks keep latency reasonable. 1024 frames at 16 kHz is ~64 ms.
    chunk_frames: int = 1024

    # Bounded queue prevents runaway RAM usage if STT is slower than recording.
    max_queue_chunks: int = 64

    # How often the optional amplitude callback should fire.
    amplitude_interval_s: float = 0.5

    # For speech-triggered recording.
    speech_threshold: float = 0.0025
    silence_threshold: float = 0.0015
    silence_timeout_s: float = 1.0
    max_record_seconds: float = 12.0
    pre_roll_seconds: float = 0.4

    def validate(self) -> bool:
        if self.sample_rate <= 0:
            raise AudioServiceError(f"sample_rate must be positive: {self.sample_rate}")
        if self.channels < 1:
            raise AudioServiceError(f"channels must be >= 1: {self.channels}")
        if self.sample_width_bytes not in (1, 2):
            raise AudioServiceError(
                f"sample_width_bytes must be 1 or 2: {self.sample_width_bytes}"
            )
        return True


@dataclass(frozen=True)
class PcmChunk:
    data: bytes
    timestamp: float
    amplitude: float


class AudioService(Service):
    """
    Non-blocking audio capture service.

    The capture thread only reads PCM and computes amplitude — it stays cheap.
    Heavy work like whisper.cpp must run in a different process/thread.
    """

    def __init__(self, config: AudioConfig = AudioConfig()) -> None:
        self.config = config
        self._proc: Optional[subprocess.Popen] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._initialized = False

        self._chunk_queue: Queue[PcmChunk] = Queue(maxsize=config.max_queue_chunks)
        self._latest_amplitude = 0.0
        self._amplitude_callback: Optional[Callable[[float], None]] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.config.validate()
        with self._lock:
            if self._initialized:
                return
            self._initialized = True

    def shutdown(self) -> None:
        self.stop_capture()
        self._initialized = False

    def reset(self) -> bool:
        self.shutdown()
        self.initialize()
        return True

    def status(self) -> dict:
        capturing = self._capture_thread is not None and self._capture_thread.is_alive()
        return {
            "enabled": self.enabled,
            "service": "ok" if self._initialized else "uninitialized",
            "capturing": capturing,
            "queue_size": self._chunk_queue.qsize(),
            "latest_amplitude": round(self._latest_amplitude, 4),
        }

    def start_capture(self, amplitude_callback: Optional[Callable[[float], None]] = None) -> None:
        if not self._initialized:
            raise AudioServiceError("AudioService.initialize() must be called first")

        with self._lock:
            if self._capture_thread and self._capture_thread.is_alive():
                return

            self._amplitude_callback = amplitude_callback
            self._stop_event.clear()
            self._clear_queue()

            cmd = self._build_arecord_command()
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="AudioCaptureThread",
                daemon=True,
            )
            self._capture_thread.start()

    def stop_capture(self) -> None:
        self._stop_event.set()

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self._terminate_arecord()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_chunk(self, timeout: Optional[float] = None) -> Optional[PcmChunk]:
        """
        Return the next PCM chunk, or None if no chunk is available before timeout.

        This is intended for an STT worker that consumes audio independently.
        """
        try:
            return self._chunk_queue.get(timeout=timeout)
        except Empty:
            return None

    def latest_amplitude(self) -> float:
        """
        Normalized mean absolute amplitude, roughly 0.0 to 1.0.
        """
        return self._latest_amplitude

    def record_fixed_wav(self, output_path: str | Path, duration_s: float) -> Path:
        """
        Blocking helper for simple tests.
        Records from the already-running capture stream into a WAV file.
        """
        output_path = Path(output_path)
        deadline = time.monotonic() + duration_s
        chunks: list[bytes] = []

        while time.monotonic() < deadline:
            chunk = self.get_chunk(timeout=0.2)
            if chunk is not None:
                chunks.append(chunk.data)

        self._write_wav(output_path, chunks)
        return output_path

    def record_until_silence(self, output_path: str | Path) -> Optional[Path]:
        """
        Blocking helper that waits for speech, records it, then stops after silence.

        Returns the written WAV path, or None if no speech was detected before max time.
        """
        output_path = Path(output_path)
        cfg = self.config

        pre_roll_max_chunks = max(
            1,
            int((cfg.pre_roll_seconds * cfg.sample_rate) / cfg.chunk_frames),
        )
        pre_roll: deque[bytes] = deque(maxlen=pre_roll_max_chunks)

        started = False
        chunks: list[bytes] = []
        first_speech_time = 0.0
        last_loud_time = 0.0
        start_wait_time = time.monotonic()

        while True:
            now = time.monotonic()

            if started and now - first_speech_time > cfg.max_record_seconds:
                break

            if not started and now - start_wait_time > cfg.max_record_seconds:
                return None

            chunk = self.get_chunk(timeout=0.2)
            if chunk is None:
                continue

            pre_roll.append(chunk.data)

            if not started:
                if chunk.amplitude >= cfg.speech_threshold:
                    started = True
                    first_speech_time = now
                    last_loud_time = now
                    chunks.extend(pre_roll)
                continue

            chunks.append(chunk.data)

            if chunk.amplitude >= cfg.silence_threshold:
                last_loud_time = now
            elif now - last_loud_time >= cfg.silence_timeout_s:
                break

        if not chunks:
            return None

        self._write_wav(output_path, chunks)
        return output_path

    # ------------------------------------------------------------------
    # Internal capture
    # ------------------------------------------------------------------
    def _build_arecord_command(self) -> list[str]:
        cfg = self.config
        return [
            "arecord",
            "-D", cfg.device,
            "-q",
            "-t", "raw",
            "-f", "S16_LE",
            "-r", str(cfg.sample_rate),
            "-c", str(cfg.channels),
        ]

    def _capture_loop(self) -> None:
        cfg = self.config
        bytes_per_chunk = cfg.chunk_frames * cfg.channels * cfg.sample_width_bytes
        last_amp_callback = time.monotonic()

        if self._proc is None or self._proc.stdout is None:
            return

        while not self._stop_event.is_set():
            data = self._proc.stdout.read(bytes_per_chunk)
            if not data:
                err = self._read_stderr()
                if err:
                    print(f"[AudioService] arecord stopped: {err}", file=sys.stderr)
                break

            amplitude = self._mean_absolute_amplitude(data)
            self._latest_amplitude = amplitude

            chunk = PcmChunk(data=data, timestamp=time.monotonic(), amplitude=amplitude)
            self._put_chunk_drop_oldest(chunk)

            now = time.monotonic()
            if self._amplitude_callback and now - last_amp_callback >= cfg.amplitude_interval_s:
                try:
                    self._amplitude_callback(amplitude)
                except Exception as exc:
                    print(f"[AudioService] amplitude callback failed: {exc}", file=sys.stderr)
                last_amp_callback = now

    def _put_chunk_drop_oldest(self, chunk: PcmChunk) -> None:
        """
        Keep capture real-time. If STT falls behind, drop old audio instead of
        blocking the capture thread and growing latency forever.
        """
        try:
            self._chunk_queue.put_nowait(chunk)
        except Full:
            try:
                self._chunk_queue.get_nowait()
            except Empty:
                pass
            try:
                self._chunk_queue.put_nowait(chunk)
            except Full:
                pass

    def _terminate_arecord(self) -> None:
        proc = self._proc
        self._proc = None

        if proc is None:
            return

        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _read_stderr(self) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        try:
            return self._proc.stderr.read().decode(errors="ignore").strip()
        except Exception:
            return ""

    def _clear_queue(self) -> None:
        while True:
            try:
                self._chunk_queue.get_nowait()
            except Empty:
                break

    # ------------------------------------------------------------------
    # PCM helpers
    # ------------------------------------------------------------------
    def _mean_absolute_amplitude(self, data: bytes) -> float:
        """
        Mean absolute amplitude for signed 16-bit little-endian PCM.
        Returns normalized amplitude from about 0.0 to 1.0.
        """
        if len(data) < 2:
            return 0.0

        total = 0
        count = 0

        # If channels > 1, this still computes average across all samples.
        for i in range(0, len(data) - 1, 2):
            sample = int.from_bytes(data[i:i + 2], byteorder="little", signed=True)
            total += abs(sample)
            count += 1

        if count == 0:
            return 0.0

        return (total / count) / 32768.0

    def _write_wav(self, output_path: Path, chunks: list[bytes]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(self.config.channels)
            wav_file.setsampwidth(self.config.sample_width_bytes)
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(b"".join(chunks))
