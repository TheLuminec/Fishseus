"""
tts_service.py

Thin local TTS service for Fishseus using Piper.

Responsibilities:
- Generate WAV speech locally with Piper
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
    wav = tts.synthesize("Behold, I awaken.")
    tts.play_wav(wav)

Or:
    tts.speak("Hello Caleb")
"""

from __future__ import annotations

import subprocess
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

    # Piper synthesis timeout
    timeout_s: float = 30.0


class TtsService:
    def __init__(self, config: TtsConfig = TtsConfig()) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_voice = self.config.default_voice

    # ------------------------------------------------------------------
    # Voice management
    # ------------------------------------------------------------------
    def available_voices(self) -> list[str]:
        if not self.config.voices_dir.exists():
            return []

        voices = []
        for voice in self.config.voices_dir.glob("*.onnx"):
            voices.append(voice.stem)

        return sorted(voices)

    def set_voice(self, voice_name: str) -> None:
        if not self._voice_path(voice_name).exists():
            raise TtsServiceError(f"Voice not found: {voice_name}")

        self.current_voice = voice_name

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
        model_path = self._voice_path(voice_name)
        config_path = self._voice_config_path(voice_name)

        if not model_path.exists():
            raise TtsServiceError(f"Voice model not found: {model_path}")

        if not config_path.exists():
            raise TtsServiceError(f"Voice config not found: {config_path}")

        if output_path is None:
            timestamp = int(time.time() * 1000)
            output_path = self.config.output_dir / f"tts_{timestamp}.wav"
        else:
            output_path = Path(output_path)

        cmd = [
            self.config.piper_binary,
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

        cmd = [
            "aplay",
            "-D", self.config.audio_device,
            str(wav_path),
        ]

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
    # Helpers
    # ------------------------------------------------------------------
    def _voice_path(self, voice_name: str) -> Path:
        return self.config.voices_dir / f"{voice_name}.onnx"

    def _voice_config_path(self, voice_name: str) -> Path:
        return self.config.voices_dir / f"{voice_name}.onnx.json"


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
        )
    )

    print("Available voices:")
    for voice in tts.available_voices():
        print(f"  - {voice}")

    print("\nTesting speech...")
    wav = tts.synthesize("Behold. Fishseus has found its voice.")
    print(f"Generated: {wav}")
    tts.play_wav(wav)
