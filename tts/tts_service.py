"""
tts_service.py

Thin local TTS service for Fishseus using Piper.

Responsibilities:
- Generate WAV speech locally with Piper
- Keep Piper running as a persistent daemon so the voice model stays in memory
- Allow voice switching
- Play WAV output through aplay / ALSA
- Keep a simple API for the orchestrator

Example:
    tts = TtsService()
    tts.initialize()           # starts the persistent piper daemon
    wav = tts.synthesize("Behold, I awaken.")
    tts.play_wav(wav)
    tts.shutdown()

Orchestrator usage:
    tts = TtsService(TtsConfig(**config.get("tts", {})))
    tts.initialize()
    tts.speak("Hello, world!") # Uses default voice and plays the WAV
    tts.speak("Hello, world!", voice="some_other_voice") # Switches voice for this line only
    tts.set_voice("some_other_voice") # Switches voice for all future speech
    tts.speak("Hello, world!") # Uses the new default voice
    tts.reset()                # Restarts the daemon with the current voice (use if broken)
    tts.shutdown()  # Cleans up daemon
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from services import Service, ServiceConfig, ServiceError, ROOT_DIR

class TtsServiceError(ServiceError):
    pass


@dataclass(frozen=True)
class TtsConfig(ServiceConfig):
    module_name: str = "tts"

    # Piper executable (absolute path into the tts venv)
    piper_binary: Path = ROOT_DIR / "tts" / ".venv" / "bin" / "piper"

    # Folder containing *.onnx voices
    voices_dir: Path = ROOT_DIR / "tts" / "voices"

    # Default voice filename stem (without .onnx)
    default_voice: str = "en_US-arctic-medium"

    # ALSA playback target
    audio_device: str = "plughw:0,0"

    # Temporary output directory
    output_dir: Path = ROOT_DIR / "tmp" / "tts"

    # Piper synthesis timeout (applies to both daemon and subprocess modes)
    timeout_s: float = 30.0

    # Keep piper alive between calls so the ONNX model stays in memory.
    # Set to False to fall back to one-subprocess-per-call behaviour.
    persistent: bool = True

    def validate(self) -> bool:
        if not self.piper_binary.exists():
            raise TtsServiceError(f"Piper binary not found: {self.piper_binary}")
        if not self.voices_dir.exists():
            raise TtsServiceError(f"Voices directory not found: {self.voices_dir}")
        if not (self.voices_dir / f"{self.default_voice}.onnx").exists():
            raise TtsServiceError(f"Default voice model not found: {self.default_voice}")
        return True


class TtsService(Service):
    def __init__(self, config: TtsConfig = TtsConfig()) -> None:
        self.config = config
        self.current_voice = config.default_voice

        # Persistent piper daemon state
        self._daemon_proc: Optional[subprocess.Popen] = None
        self._daemon_voice: Optional[str] = None
        # piper (>=1.4) picks its own timestamped filenames in --output-dir
        # mode, so the daemon writes here and we move each WAV to its final
        # path.  Kept separate from output_dir to avoid clashing with the
        # subprocess-mode files.
        self._daemon_out_dir = self.config.output_dir / "_daemon"
        # Reentrant: public methods take the lock and call into daemon
        # helpers that take it again on the same thread.
        self._daemon_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Start the persistent Piper daemon (if persistent=True)."""
        self.config.validate()  # raises TtsServiceError on any problem

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._daemon_out_dir.mkdir(parents=True, exist_ok=True)
        self.current_voice = self.config.default_voice

        if self.config.persistent:
            with self._daemon_lock:
                self._start_daemon(self.current_voice)

    def shutdown(self) -> None:
        """Terminate the daemon process cleanly."""
        self._stop_daemon()

    def status(self) -> dict:
        """Return a dict of status information."""
        return {
            "enabled": self.enabled,
            "service": "ok",
            "daemon_alive": self._daemon_alive(),
            "current_voice": self.current_voice,
        }

    def reset(self) -> bool:
        """Reset the service.  Return True if successful, False otherwise."""
        self._stop_daemon()
        if self.config.persistent:
            with self._daemon_lock:
                self._start_daemon(self.current_voice)

            return self._daemon_alive()

        return True
        
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
        self.reset()

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse all whitespace (including newlines) into single spaces.

        Piper reads ONE line of text per synthesis and writes one WAV per line.
        If an utterance contains embedded newlines, piper emits several WAVs and
        only the first is collected — so a multi-sentence reply would lose every
        sentence after the first. Flattening to a single line keeps the whole
        utterance in one WAV; sentence punctuation still drives natural pauses.
        """
        return " ".join((text or "").split())

    def synthesize(
        self,
        text: str,
        output_path: Optional[str | Path] = None,
        voice: Optional[str] = None,
    ) -> Path:
        text = self._normalize_text(text)
        if not text:
            raise TtsServiceError("Cannot synthesize empty text")

        voice_name = voice or self.current_voice

        if output_path is None:
            timestamp = int(time.time() * 1000)
            output_path = self.config.output_dir / f"tts_{timestamp}.wav"
        output_path = Path(output_path)

        if output_path.exists():
            output_path.unlink()

        # Use the resident daemon when possible — avoids reloading the ONNX model.
        if self.config.persistent and voice_name == self._daemon_voice:
            if self._daemon_alive():
                try:
                    return self._synthesize_daemon(text, output_path)
                except TtsServiceError as exc:
                    print(f"[TtsService] Daemon synthesis failed ({exc}), restarting daemon…")
                    self.reset()
                    # One retry after daemon restart; fall through to the
                    # subprocess path if the daemon did not come back up.
                    if self._daemon_alive():
                        return self._synthesize_daemon(text, output_path)

        return self._synthesize_subprocess(text, output_path, voice_name)

    def _synthesize_daemon(self, text: str, output_path: Path) -> Path:
        """Speak one line to the resident piper process and collect its WAV.

        piper (>=1.4) has no JSON protocol.  In --output-dir mode it stays
        resident, reading one line of text per synthesis from stdin and
        writing a timestamped WAV into the daemon output dir.  We clear the
        dir, send the line, wait for the new file to appear and finish
        writing, then move it to output_path.
        """
        proc = self._daemon_proc

        with self._daemon_lock:
            # Empty the dir so the only WAV that shows up is this call's.
            for stale in self._daemon_out_dir.glob("*.wav"):
                stale.unlink(missing_ok=True)

            try:
                proc.stdin.write(text + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise TtsServiceError(
                    f"Write to piper daemon failed: {exc}{self._daemon_exit_info(proc)}"
                ) from exc

            produced = self._await_daemon_wav(proc)
            produced.replace(output_path)

        return output_path

    def _await_daemon_wav(self, proc: subprocess.Popen) -> Path:
        """Wait for piper to write a WAV into the (pre-cleared) output dir and
        finish flushing it.  Returns the path of that file.
        """
        deadline = time.monotonic() + self.config.timeout_s
        candidate: Optional[Path] = None
        last_size = -1
        stable = 0

        while True:
            if candidate is None:
                found = sorted(self._daemon_out_dir.glob("*.wav"))
                if found:
                    candidate = found[0]
            else:
                try:
                    size = candidate.stat().st_size
                except FileNotFoundError:
                    candidate, size = None, -1
                # A WAV header is 44 bytes; wait for the size to settle so we
                # don't move the file out from under piper mid-write.
                if candidate is not None and size == last_size and size > 44:
                    stable += 1
                    if stable >= 2:
                        return candidate
                else:
                    stable = 0
                last_size = size

            if time.monotonic() > deadline:
                raise TtsServiceError("Piper daemon synthesis timed out")
            if proc.poll() is not None:
                raise TtsServiceError(
                    f"Piper daemon exited unexpectedly during synthesis"
                    f"{self._daemon_exit_info(proc)}"
                )
            time.sleep(0.02)

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
        """Synthesize and play the given text.  Return the WAV path."""
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
            raise TtsServiceError(f"[TtsService] Cannot start daemon: voice files missing for '{voice_name}'")
            return

        # Fresh output dir so leftover WAVs can't be mistaken for new output.
        self._daemon_out_dir.mkdir(parents=True, exist_ok=True)
        for stale in self._daemon_out_dir.glob("*.wav"):
            stale.unlink(missing_ok=True)

        # piper stays resident in --output-dir mode: one line of text on stdin
        # per synthesis, one timestamped WAV written to the output dir.
        cmd = [
            str(self.config.piper_binary),
            "--model", str(model_path),
            "--config", str(config_path),
            "--output-dir", str(self._daemon_out_dir),
            "--output-dir-naming", "timestamp",
        ]
        with self._daemon_lock:
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

    def _daemon_exit_info(self, proc: Optional[subprocess.Popen]) -> str:
        """Best-effort exit code + stderr from a daemon that has already died.

        Only reads stderr once the process has actually exited, otherwise the
        read would block on the still-open pipe.
        """
        if proc is None:
            return ""
        rc = proc.poll()
        if rc is None:
            return ""
        try:
            err = proc.stderr.read() if proc.stderr else ""
        except Exception:
            err = ""
        detail = f" (exit code {rc})"
        if err and err.strip():
            detail += f"\n[piper stderr]\n{err.strip()}"
        return detail

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _voice_path(self, voice_name: str) -> Path:
        return self.config.voices_dir / f"{voice_name}.onnx"

    def _voice_config_path(self, voice_name: str) -> Path:
        return self.config.voices_dir / f"{voice_name}.onnx.json"

