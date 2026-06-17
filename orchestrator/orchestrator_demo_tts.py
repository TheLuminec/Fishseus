"""
orchestrator_demo_with_tts.py

TTS-enabled demo orchestrator for the Fishseus/Billy Bass assistant project.

Flow:

    USB mic
      -> audio_service.record_until_silence(...)
      -> stt_service.transcribe_file(...)
      -> wake-word check
      -> assistant_service.handle_user_text(...)
      -> tts_service.synthesize(...)
      -> play WAV through HiFiBerry / aplay
      -> motion_service.speak_audio(WAV)

This version assumes:
- audio_service.py works with your USB mic
- stt_service.py works with whisper.cpp
- llm_service.py works against your remote Ollama/OpenAI-compatible endpoint
- assistant_service.py works
- tts_service.py works with Piper
- motion_service.py works with your L298N motor setup

Expected project structure example:

    fishseus/
      audio/audio_service.py
      stt/stt_service.py
      llm/llm_service.py
      assistant/assistant_service.py
      motion/motion_service.py
      tts/tts_service.py
      orchestrator/orchestrator_demo_with_tts.py
      data/
      config/
      tmp/
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------------------
# Import path setup
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for subdir in ["audio", "stt", "llm", "assistant", "motion", "tts"]:
    candidate = PROJECT_ROOT / subdir
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from audio_service import AudioConfig, AudioService
from stt_service import SttConfig, SttService
from llm_service import LlmConfig, LlmService
from assistant_service import AssistantConfig, AssistantService, Tool, ToolRegistry
from motion_service import MotionService, MotorConfig
from tts_service import TtsConfig, TtsService


# ----------------------------------------------------------------------
# Demo configuration
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class OrchestratorConfig:
    # Audio input
    audio_device: str = "plughw:3,0"
    speech_wav_path: Path = PROJECT_ROOT / "tmp" / "latest_speech.wav"

    # STT / whisper.cpp
    whisper_binary: Path = PROJECT_ROOT / "stt" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    whisper_model: Path = PROJECT_ROOT / "stt" / "whisper.cpp" / "models" / "ggml-base.en.bin"
    whisper_threads: int = 4
    wake_words: tuple[str, ...] = ("fish", "fishseus", "hey fish")

    # Remote LLM server
    llm_endpoint: str = "http://ollama.angelfish-gamma.ts.net/v1/chat/completions"
    llm_model: str = "qwen2.5:3b"

    # Assistant config files
    personality_path: Path = PROJECT_ROOT / "config" / "personality_prompt.txt"
    memory_path: Path = PROJECT_ROOT / "data" / "assistant_memory.json"
    history_path: Path = PROJECT_ROOT / "data" / "conversation_log.jsonl"

    # TTS / Piper
    piper_binary: str = PROJECT_ROOT / "tts" / ".venv" / "bin" / "piper"
    voices_dir: Path = PROJECT_ROOT / "tts" / "voices"
    default_voice: str = "en_US-arctic-medium"
    tts_audio_device: str = "plughw:0,0"
    tts_output_dir: Path = PROJECT_ROOT / "tmp" / "tts"

    # General loop behavior
    print_ignored_transcripts: bool = True
    no_speech_sleep_s: float = 0.05

    # If true, fish mouth animation and aplay are started at the same time.
    # This is what you want for normal operation.
    sync_motion_with_tts: bool = True


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

class FishseusTtsDemoOrchestrator:
    def __init__(self, config: OrchestratorConfig = OrchestratorConfig()) -> None:
        self.config = config
        self.running = False

        self.audio: Optional[AudioService] = None
        self.stt: Optional[SttService] = None
        self.llm: Optional[LlmService] = None
        self.motion: Optional[MotionService] = None
        self.tts: Optional[TtsService] = None
        self.assistant: Optional[AssistantService] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        config_file = PROJECT_ROOT / "config" / "fish_config.json"
        app_config = {}
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                app_config = json.load(f)

        audio_cfg = app_config.get("audio", {})
        stt_cfg = app_config.get("stt", {})
        llm_cfg = app_config.get("llm", {})
        tts_cfg = app_config.get("tts", {})
        motion_cfg = app_config.get("motion", {})
        assistant_cfg = app_config.get("assistant", {})

        cfg = self.config
        cfg.speech_wav_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.memory_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Resolve paths relative to config dir if they are relative
        config_dir = config_file.parent
        pers_path = (config_dir / assistant_cfg.get("personality_path", "personality_prompt.txt")).resolve()
        pers_path.parent.mkdir(parents=True, exist_ok=True)

        tts_out = (config_dir / tts_cfg.get("output_dir", "../tmp/tts")).resolve()
        tts_out.mkdir(parents=True, exist_ok=True)

        print("[orchestrator] Initializing audio service...")
        self.audio = AudioService(
            AudioConfig(
                device=audio_cfg.get("device", cfg.audio_device),
                sample_rate=audio_cfg.get("sample_rate", 16000),
                channels=audio_cfg.get("channels", 1),
                speech_threshold=audio_cfg.get("speech_threshold", 0.003),
                silence_threshold=audio_cfg.get("silence_threshold", 0.0008),
                silence_timeout_s=audio_cfg.get("silence_timeout_s", 1.0),
                max_record_seconds=audio_cfg.get("max_record_seconds", 12.0),
                pre_roll_seconds=audio_cfg.get("pre_roll_seconds", 0.4),
            )
        )
        self.audio.initialize()

        print("[orchestrator] Initializing STT service...")
        self.stt = SttService(
            SttConfig(
                whisper_binary=str((config_dir / stt_cfg.get("whisper_binary", "../stt/whisper.cpp/build/bin/whisper-cli")).resolve()),
                model_path=str((config_dir / stt_cfg.get("model_path", "../stt/whisper.cpp/models/ggml-base.en.bin")).resolve()),
                threads=stt_cfg.get("threads", cfg.whisper_threads),
                wake_words=stt_cfg.get("wake_words", list(cfg.wake_words)),
                strip_wake_word=False,
            )
        )
        self.stt.initialize()

        print("[orchestrator] Initializing LLM service...")
        self.llm = LlmService(
            LlmConfig(
                endpoint_url=llm_cfg.get("endpoint_url", cfg.llm_endpoint),
                model=llm_cfg.get("model", cfg.llm_model),
                timeout_s=45.0,
                retries=1,
                temperature=llm_cfg.get("temperature", 0.7),
                max_tokens=llm_cfg.get("max_tokens", 512),
                disable_reasoning=llm_cfg.get("disable_reasoning", True),
            )
        )

        print("[orchestrator] Initializing TTS service...")
        self.tts = TtsService(
            TtsConfig(
                piper_binary=str((config_dir / tts_cfg.get("piper_binary", "../tts/.venv/bin/piper")).resolve()),
                voices_dir=Path((config_dir / tts_cfg.get("voices_dir", "../tts/voices")).resolve()),
                default_voice=tts_cfg.get("default_voice", cfg.default_voice),
                audio_device=tts_cfg.get("audio_device", cfg.tts_audio_device),
                output_dir=tts_out,
            )
        )
        voices = self.tts.available_voices()
        print(f"[orchestrator] Available TTS voices: {voices}")

        print("[orchestrator] Initializing motion service...")
        self.motion = self._build_motion_service(motion_cfg)
        self.motion.initialize()

        print("[orchestrator] Initializing assistant service...")
        tools = self._build_tool_registry(self.motion, self.tts)
        self.assistant = AssistantService(
            llm=self.llm,
            config=AssistantConfig(
                assistant_name=assistant_cfg.get("assistant_name", "Fishseus"),
                user_name=assistant_cfg.get("user_name", "User"),
                personality_path=pers_path,
                memory_path=cfg.memory_path,
                history_path=cfg.history_path,
            ),
            tool_registry=tools,
        )

        print("[orchestrator] Starting audio capture...")
        self.audio.start_capture(amplitude_callback=self._print_amplitude_debug)
        print("[orchestrator] Ready. Say the wake word, e.g. 'fish'. Press Ctrl+C to stop.")

    def _build_motion_service(self, motion_cfg: dict) -> MotionService:
        """
        Build motor configs dynamically from JSON or fallback to defaults.
        """
        motors_raw = motion_cfg.get("motors", {})
        motors = {}
        for name, m in motors_raw.items():
            motors[name] = MotorConfig(
                in1=int(m.get("in1", 0)),
                in2=int(m.get("in2", 0)),
                en=int(m.get("en", 0)),
                forward_speed=float(m.get("forward_speed", 70)),
                reverse_speed=float(m.get("reverse_speed", 55)),
                neutral_return_time=float(m.get("neutral_return_time", 0.08)),
            )

        if not motors:
            motors = {
                "mouth": MotorConfig(17, 27, 22, 82, 55, 0.04),
                "tail": MotorConfig(23, 24, 25, 72, 48, 0.03),
                "body": MotorConfig(5, 6, 12, 68, 45, 0.03),
            }

        return MotionService(
            motors=motors,
            pwm_frequency=int(motion_cfg.get("pwm_frequency", 1000)),
            body_wiggle_time=float(motion_cfg.get("body_wiggle_time", 0.18)),
            tail_wiggle_time=float(motion_cfg.get("tail_wiggle_time", 0.14)),
            mouth_open_time=float(motion_cfg.get("mouth_open_time", 0.09)),
            mouth_close_time=float(motion_cfg.get("mouth_close_time", 0.04)),
            envelope_window_s=float(motion_cfg.get("envelope_window_s", 0.18)),
        )

    def _build_tool_registry(self, motion: MotionService, tts: TtsService) -> ToolRegistry:
        registry = ToolRegistry()

        def wiggle(cycles: int = 1) -> str:
            cycles = max(1, min(int(cycles), 5))
            motion.wiggle(cycles=cycles)
            return f"wiggle queued for {cycles} cycle(s)"

        def open_mouth() -> str:
            motion.open_mouth()
            return "mouth open queued"

        def set_mode(mode: str) -> str:
            allowed = {"assistant", "bluetooth"}
            if mode not in allowed:
                raise ValueError(f"mode must be one of {sorted(allowed)}")
            # In this demo, mode switching is not implemented yet.
            return f"mode switch requested: {mode}"

        def get_current_time() -> str:
            return time.strftime("%Y-%m-%d %H:%M:%S")

        def list_voices() -> str:
            voices = tts.available_voices()
            return ", ".join(voices) if voices else "no voices found"

        def set_voice(voice: str) -> str:
            tts.set_voice(voice)
            return f"voice set to {voice}"

        registry.register(
            Tool(
                name="wiggle",
                description="Make the fish wiggle. Args: cycles integer from 1 to 5.",
                function=wiggle,
                risk="safe",
            )
        )
        registry.register(
            Tool(
                name="open_mouth",
                description="Open the fish mouth once. Args: none.",
                function=open_mouth,
                risk="safe",
            )
        )
        registry.register(
            Tool(
                name="set_mode",
                description="Request a mode switch. Args: mode is 'assistant' or 'bluetooth'.",
                function=set_mode,
                risk="safe",
            )
        )
        registry.register(
            Tool(
                name="get_current_time",
                description="Get the current system time. Args: none.",
                function=get_current_time,
                risk="safe",
            )
        )
        registry.register(
            Tool(
                name="list_voices",
                description="List available Piper TTS voices. Args: none.",
                function=list_voices,
                risk="safe",
            )
        )
        registry.register(
            Tool(
                name="set_voice",
                description="Set the current TTS voice. Args: voice is an available voice name.",
                function=set_voice,
                risk="safe",
            )
        )

        return registry

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        if not all([self.audio, self.stt, self.assistant, self.motion, self.tts]):
            raise RuntimeError("initialize() must be called before run()")

        self.running = True

        while self.running:
            print("\n[orchestrator] Listening for speech...")
            wav_path = self.audio.record_until_silence(self.config.speech_wav_path)

            if not wav_path:
                time.sleep(self.config.no_speech_sleep_s)
                continue

            print(f"[orchestrator] Captured speech: {wav_path}")

            try:
                stt_result = self.stt.transcribe_file(wav_path)
            except Exception as exc:
                print(f"[orchestrator] STT failed: {exc}")
                self.motion.wiggle(cycles=1)
                continue

            transcript = stt_result.text.strip()
            print(f"[stt] {transcript}")
            print(f"[stt] wake_detected={stt_result.wake_detected}, command='{stt_result.command_text}'")

            if not stt_result.wake_detected:
                if self.config.print_ignored_transcripts:
                    print("[orchestrator] Wake word not detected. Ignoring.")
                continue

            command_text = stt_result.command_text.strip()
            if not command_text:
                command_text = "Hello."

            self._handle_wake_command(command_text)

    def _handle_wake_command(self, command_text: str) -> None:
        assert self.assistant is not None
        assert self.motion is not None
        assert self.tts is not None

        print(f"[orchestrator] Handling command: {command_text}")

        # Immediate feedback so the fish feels responsive while the LLM thinks.
        self.motion.wiggle(cycles=1)

        try:
            result = self.assistant.handle_user_text(command_text)
        except Exception as exc:
            print(f"[orchestrator] Assistant failed: {exc}")
            self.motion.speak_text_placeholder(duration_s=1.0)
            return

        print("\n--- Fishseus ---")
        print(f"Speak:       {result.speak}")
        print(f"Motion:      {result.motion}")
        print(f"Tool calls:  {[call.__dict__ for call in result.tool_calls]}")
        print(f"Tool results:{result.tool_results}")
        print(f"Memory:      {result.memory_updates}")
        print(f"Elapsed:     {result.elapsed_s:.2f}s")
        print("----------------\n")

        self._speak_with_motion(result.speak, result.motion)

    def _speak_with_motion(self, text: str, motion_hint: str) -> None:
        assert self.tts is not None
        assert self.motion is not None

        try:
            print("[tts] Synthesizing response...")
            tts_wav = self.tts.synthesize(text)
            print(f"[tts] Generated {tts_wav}")
        except Exception as exc:
            print(f"[tts] Synthesis failed: {exc}")
            self.motion.speak_text_placeholder(duration_s=1.5)
            return

        # Extra reaction before speech starts.
        if motion_hint in {"happy", "excited"}:
            self.motion.wiggle(cycles=1)
        elif motion_hint == "annoyed":
            self.motion.open_mouth()
        elif motion_hint == "thinking":
            self.motion.wiggle(cycles=1, tail=True, body=False)

        if self.config.sync_motion_with_tts:
            # Start motion and audio at nearly the same time.
            # motion.speak_audio is non-blocking because it queues work to the
            # motion worker, while play_wav blocks until audio is done.
            print("[tts] Playing with synchronized mouth motion...")
            self.motion.speak_audio(tts_wav)
            try:
                self.tts.play_wav(tts_wav, blocking=True)
            except Exception as exc:
                print(f"[tts] Playback failed: {exc}")
        else:
            # Debug mode: audio first, then mouth animation.
            try:
                self.tts.play_wav(tts_wav, blocking=True)
            except Exception as exc:
                print(f"[tts] Playback failed: {exc}")
            self.motion.speak_audio(tts_wav)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self.running = False

    def shutdown(self) -> None:
        print("\n[orchestrator] Shutting down...")

        if self.audio is not None:
            try:
                self.audio.shutdown()
            except Exception as exc:
                print(f"[orchestrator] Audio shutdown error: {exc}")

        if self.stt is not None:
            try:
                self.stt.shutdown()
            except Exception as exc:
                print(f"[orchestrator] STT shutdown error: {exc}")

        if self.motion is not None:
            try:
                self.motion.stop_all()
                self.motion.shutdown()
            except Exception as exc:
                print(f"[orchestrator] Motion shutdown error: {exc}")

        if self.llm is not None:
            try:
                self.llm.close()
            except Exception as exc:
                print(f"[orchestrator] LLM shutdown error: {exc}")

        print("[orchestrator] Shutdown complete.")

    @staticmethod
    def _print_amplitude_debug(amp: float) -> None:
        # Comment this out if it is too noisy.
        print(f"[audio] amp={amp:.4f}")


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

def main() -> int:
    orchestrator = FishseusTtsDemoOrchestrator()

    def handle_signal(signum, frame):  # noqa: ANN001
        orchestrator.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        orchestrator.initialize()
        orchestrator.run()
    except KeyboardInterrupt:
        orchestrator.stop()
    finally:
        orchestrator.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
