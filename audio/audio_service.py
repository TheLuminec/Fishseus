"""
audio_service.py

Lightweight audio service for the Fishseus/Billy Bass assistant project.

Design goals:
- Use a working USB microphone through ALSA/arecord.
- Keep capture low-overhead so whisper.cpp/STT can run in parallel.
- Avoid loading large Python audio frameworks.
- Provide bounded queues so audio cannot endlessly consume RAM.
- Provide simple APIs for:
    - live amplitude monitoring
    - recording a fixed-duration WAV
    - recording speech after a threshold trigger
    - streaming PCM chunks to an STT worker

This module intentionally does not run Whisper, TTS, or the LLM.
Those should remain separate services coordinated by the orchestrator.
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


@dataclass(frozen=True)
class AudioConfig:
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


@dataclass(frozen=True)
class PcmChunk:
    data: bytes
    timestamp: float
    amplitude: float


class AudioService:
    """
    Non-blocking audio capture service.

    Typical use:

        audio = AudioService(AudioConfig(device="plughw:1,0"))
        audio.initialize()
        audio.start_capture()

        chunk = audio.get_chunk(timeout=0.1)
        # Send chunk.data to STT worker, or collect chunks into a WAV.

        audio.shutdown()

    The capture thread only reads PCM and computes amplitude. It should stay cheap.
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
        with self._lock:
            if self._initialized:
                return
            self._initialized = True

    def start_capture(self, amplitude_callback: Optional[Callable[[float], None]] = None) -> None:
        if not self._initialized:
            raise RuntimeError("AudioService.initialize() must be called first")

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

    def shutdown(self) -> None:
        self.stop_capture()
        self._initialized = False

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
        This is useful before integrating a real VAD/wake-word system.
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


# ----------------------------------------------------------------------
# Manual test runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # For USB mics, run `arecord -L` and set this to the stable device name.
    # Examples:
    #   device="default"
    #   device="plughw:1,0"
    #   device="plughw:CARD=Device,DEV=0"
    config = AudioConfig(device="default")
    audio = AudioService(config)

    def print_amp(amp: float) -> None:
        print(f"Average amplitude: {amp:.4f}")

    audio.initialize()
    audio.start_capture(amplitude_callback=print_amp)

    try:
        print("Recording 5 seconds to test_recording.wav...")
        path = audio.record_fixed_wav("test_recording.wav", duration_s=5.0)
        print(f"Saved {path}")

        print("Now waiting for speech and recording until silence...")
        path = audio.record_until_silence("speech_test.wav")
        if path:
            print(f"Saved {path}")
        else:
            print("No speech detected.")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        audio.shutdown()
