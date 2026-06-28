"""
tts_service.py

Thin local TTS service for Fishseus using Piper.

Responsibilities:
- Generate WAV speech locally with Piper
- Keep Piper running as a persistent daemon so the voice model stays in memory
- Allow voice switching
- Play WAV output through aplay / ALSA
- Keep a simple API for the orchestrator

Non-responsibilities:
- No fish motion logic
- No assistant logic
- No memory
- No LLM logic

Example:
    tts = TtsService()
    tts.initialize()           # starts the persistent piper daemon
    wav = tts.synthesize("Behold, I awaken.")
    tts.play_wav(wav)
    tts.shutdown()
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class TtsServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class TtsConfig:
    # Piper executable in PATH
    piper_binary: str = "piper"

    # Folder containing *.onnx voices
    voices_dir: Path = Path("../tts/voices")

    # Default voice filename stem (without .onnx)
    default_voice: str = "en_US-arctic-medium"

    # ALSA playback target
    audio_device: str = "plughw:0,0"

    # Temporary output directory
    output_dir: Path = Path("../tmp/tts")

    # Piper synthesis timeout (applies to both daemon and subprocess modes)
    timeout_s: float = 30.0

    # Keep piper alive between calls so the ONNX model stays in memory.
    # Set to False to fall back to one-subprocess-per-call behaviour.
    persistent: bool = True


class TtsService:
    def __init__(self, config: TtsConfig = TtsConfig()) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_voice = self.config.default_voice

        # Persistent piper daemon state
        self._daemon_proc: Optional[subprocess.Popen] = None
        self._daemon_voice: Optional[str] = None
        self._daemon_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Start the persistent Piper daemon (if persistent=True)."""
        if self.config.persistent:
            self._start_daemon(self.current_voice)

    def shutdown(self) -> None:
        """Terminate the daemon process cleanly."""
        self._stop_daemon()

    # ------------------------------------------------------------------
    # Voice management
    # ------------------------------------------------------------------

    def available_voices(self) -> list[str]:
        if not self.config.voices_dir.exists():
            return []
        return sorted(v.stem for v in self.config.voices_dir.glob("*.onnx"))

    def set_voice(self, voice_name: str) -> None:
        if not self._voice_path(voice_name).exists():
            raise TtsServiceError(f"Voice not found: {voice_name}")
        self.current_voice = voice_name
        # Restart daemon so the new model is pre-loaded.
        if self.config.persistent:
            with self._daemon_lock:
                self._start_daemon(voice_name)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        output_path: Optional[str | Path] = None,
        voice: Optional[str] = None,
    ) -> Path:
        text = text.strip()
        if not text:
            raise TtsServiceError("Cannot synthesize empty text")

        voice_name = voice or self.current_voice

        if output_path is None:
            timestamp = int(time.time() * 1000)
            output_path = self.config.output_dir / f"tts_{timestamp}.wav"
        output_path = Path(output_path)

        # Use the resident daemon when possible — avoids reloading the ONNX model.
        if self.config.persistent and voice_name == self._daemon_voice:
            with self._daemon_lock:
                if self._daemon_alive():
                    try:
                        return self._synthesize_daemon(text, output_path)
                    except TtsServiceError as exc:
                        print(f"[TtsService] Daemon synthesis failed ({exc}), restarting daemon…")
                        self._start_daemon(voice_name)
                        # One retry after daemon restart
                        if self._daemon_alive():
                            return self._synthesize_daemon(text, output_path)

        return self._synthesize_subprocess(text, output_path, voice_name)

    def _synthesize_daemon(self, text: str, output_path: Path) -> Path:
        """Send one synthesis request to the resident piper process."""
        if output_path.exists():
            output_path.unlink()

        payload = json.dumps({"text": text, "output_file": str(output_path)})
        try:
            self._daemon_proc.stdin.write(payload + "\n")
            self._daemon_proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise TtsServiceError(f"Write to piper daemon failed: {exc}") from exc

        deadline = time.monotonic() + self.config.timeout_s
        while not output_path.exists():
            if time.monotonic() > deadline:
                raise TtsServiceError("Piper daemon synthesis timed out")
            if not self._daemon_alive():
                raise TtsServiceError("Piper daemon exited unexpectedly during synthesis")
            time.sleep(0.02)

        # Brief pause to ensure piper has closed/flushed the WAV before we read it.
        time.sleep(0.05)
        return output_path

    def _synthesize_subprocess(self, text: str, output_path: Path, voice_name: str) -> Path:
        """Fallback: spawn a fresh piper process for one utterance."""
        model_path = self._voice_path(voice_name)
        config_path = self._voice_config_path(voice_name)

        if not model_path.exists():
            raise TtsServiceError(f"Voice model not found: {model_path}")
        if not config_path.exists():
            raise TtsServiceError(f"Voice config not found: {config_path}")

        cmd = [
            str(self.config.piper_binary),
            "--model", str(model_path),
            "--config", str(config_path),
            "--output_file", str(output_path),
        ]

        try:
            proc = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise TtsServiceError("Piper synthesis timed out") from exc
        except FileNotFoundError as exc:
            raise TtsServiceError(
                f"Piper binary not found: {self.config.piper_binary}"
            ) from exc

        if proc.returncode != 0:
            raise TtsServiceError(
                f"Piper failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not output_path.exists():
            raise TtsServiceError("Piper reported success but WAV was not created")

        return output_path

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play_wav(self, wav_path: str | Path, blocking: bool = True) -> None:
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise TtsServiceError(f"WAV file not found: {wav_path}")

        cmd = ["aplay", "-D", self.config.audio_device, str(wav_path)]
        try:
            if blocking:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise TtsServiceError(
                        f"aplay failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                    )
            else:
                subprocess.Popen(cmd)
        except FileNotFoundError as exc:
            raise TtsServiceError("aplay not found (install alsa-utils)") from exc

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        blocking: bool = True,
    ) -> Path:
        wav = self.synthesize(text, voice=voice)
        self.play_wav(wav, blocking=blocking)
        return wav

    # ------------------------------------------------------------------
    # Daemon management
    # ------------------------------------------------------------------

    def _start_daemon(self, voice_name: str) -> None:
        """Start a persistent piper process with the given voice loaded."""
        self._stop_daemon()

        model_path = self._voice_path(voice_name)
        config_path = self._voice_config_path(voice_name)

        if not model_path.exists() or not config_path.exists():
            print(f"[TtsService] Cannot start daemon: voice files missing for '{voice_name}'")
            return

        cmd = [
            str(self.config.piper_binary),
            "--model", str(model_path),
            "--config", str(config_path),
            "--json-input",  # read {"text":…,"output_file":…} JSON lines from stdin
        ]

        try:
            self._daemon_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
            self._daemon_voice = voice_name
            print(f"[TtsService] Piper daemon started (voice={voice_name}, pid={self._daemon_proc.pid})")
        except FileNotFoundError:
            print(f"[TtsService] Piper binary not found: {self.config.piper_binary}")
            self._daemon_proc = None
        except Exception as exc:
            print(f"[TtsService] Daemon start failed: {exc}")
            self._daemon_proc = None

    def _stop_daemon(self) -> None:
        proc = self._daemon_proc
        self._daemon_proc = None
        self._daemon_voice = None

        if proc is None:
            return

        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass

    def _daemon_alive(self) -> bool:
        return self._daemon_proc is not None and self._daemon_proc.poll() is None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _voice_path(self, voice_name: str) -> Path:
        return Path(self.config.voices_dir) / f"{voice_name}.onnx"

    def _voice_config_path(self, voice_name: str) -> Path:
        return Path(self.config.voices_dir) / f"{voice_name}.onnx.json"


# ----------------------------------------------------------------------
# Manual test runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    tts = TtsService(
        TtsConfig(
            piper_binary="piper",
            voices_dir=Path("./voices"),
            default_voice="en_US-arctic-medium",
            audio_device="plughw:0,0",
            persistent=True,
        )
    )

    print("Available voices:")
    for voice in tts.available_voices():
        print(f"  - {voice}")

    tts.initialize()
    try:
        print("\nTesting speech (daemon mode)…")
        wav = tts.synthesize("Behold. Fishseus has found its voice.")
        print(f"Generated: {wav}")
        tts.play_wav(wav)

        print("\nSecond utterance (model already in memory)…")
        wav2 = tts.synthesize("The deep sea calls. I answer.")
        print(f"Generated: {wav2}")
        tts.play_wav(wav2)
    finally:
        tts.shutdown()
