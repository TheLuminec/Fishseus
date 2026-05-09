"""
orchestrator_demo.py

Demo orchestrator for the Fishseus/Billy Bass assistant project.

Current flow, with TTS intentionally skipped:

    USB mic
      -> audio_service.record_until_silence(...)
      -> stt_service.transcribe_file(...)
      -> wake-word check
      -> assistant_service.handle_user_text(...)
      -> print assistant response
      -> execute/observe tool calls
      -> motion placeholder animation

This file is intentionally simple and procedural. It proves the complete control
loop before introducing TTS, Bluetooth mode, MCP, or service supervision.

Expected project structure example:

    fishseus/
      audio/audio_service.py
      stt/stt_service.py
      llm/llm_service.py
      assistant/assistant_service.py
      motion/motion_service.py
      orchestrator/orchestrator_demo.py
      data/
      config/
      tmp/

If your folder layout differs, adjust the sys.path setup below.
"""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------------------
# Import path setup
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for subdir in ["audio", "stt", "llm", "assistant", "motion"]:
    candidate = PROJECT_ROOT / subdir
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from audio_service import AudioConfig, AudioService
from stt_service import SttConfig, SttService
from llm_service import LlmConfig, LlmService
from assistant_service import AssistantConfig, AssistantService, Tool, ToolRegistry
from motion_service import MotionService, MotorConfig


# ----------------------------------------------------------------------
# Demo configuration
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class OrchestratorConfig:
    # Audio
    audio_device: str = "default"
    speech_wav_path: Path = PROJECT_ROOT / "tmp" / "latest_speech.wav"

    # STT / whisper.cpp
    whisper_binary: Path = PROJECT_ROOT / "stt" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
    whisper_model: Path = PROJECT_ROOT / "stt" / "whisper.cpp" / "models" / "ggml-base.en.bin"
    whisper_threads: int = 4
    wake_words: tuple[str, ...] = ("fish", "fishseus", "hey fish")

    # Remote LLM server
    llm_endpoint: str = "http://ollama.angelfish-gamma.ts.net/v1/chat/completions"
    llm_model: str = "gemma4:e2b"

    # Assistant config files
    personality_path: Path = PROJECT_ROOT / "config" / "personality_prompt.txt"
    memory_path: Path = PROJECT_ROOT / "data" / "assistant_memory.json"
    history_path: Path = PROJECT_ROOT / "data" / "conversation_log.jsonl"

    # General loop behavior
    print_ignored_transcripts: bool = True
    no_speech_sleep_s: float = 0.05


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

class FishseusDemoOrchestrator:
    def __init__(self, config: OrchestratorConfig = OrchestratorConfig()) -> None:
        self.config = config
        self.running = False

        self.audio: Optional[AudioService] = None
        self.stt: Optional[SttService] = None
        self.llm: Optional[LlmService] = None
        self.motion: Optional[MotionService] = None
        self.assistant: Optional[AssistantService] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        cfg = self.config
        cfg.speech_wav_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.memory_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.history_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.personality_path.parent.mkdir(parents=True, exist_ok=True)

        print("[orchestrator] Initializing audio service...")
        self.audio = AudioService(
            AudioConfig(
                device=cfg.audio_device,
                sample_rate=16000,
                channels=1,
                speech_threshold=0.0025,
                silence_threshold=0.0015,
                silence_timeout_s=1.0,
                max_record_seconds=12.0,
                pre_roll_seconds=0.4,
            )
        )
        self.audio.initialize()

        print("[orchestrator] Initializing STT service...")
        self.stt = SttService(
            SttConfig(
                whisper_binary=str(cfg.whisper_binary),
                model_path=str(cfg.whisper_model),
                threads=cfg.whisper_threads,
                wake_words=cfg.wake_words,
                strip_wake_word=True,
            )
        )
        self.stt.initialize()

        print("[orchestrator] Initializing LLM service...")
        self.llm = LlmService(
            LlmConfig(
                endpoint_url=cfg.llm_endpoint,
                model=cfg.llm_model,
                timeout_s=45.0,
                retries=1,
                temperature=0.7,
                max_tokens=350,
                disable_reasoning=True,
            )
        )

        print("[orchestrator] Initializing motion service...")
        self.motion = self._build_motion_service()
        self.motion.initialize()

        print("[orchestrator] Initializing assistant service...")
        tools = self._build_tool_registry(self.motion)
        self.assistant = AssistantService(
            llm=self.llm,
            config=AssistantConfig(
                personality_path=cfg.personality_path,
                memory_path=cfg.memory_path,
                history_path=cfg.history_path,
            ),
            tool_registry=tools,
        )

        print("[orchestrator] Starting audio capture...")
        self.audio.start_capture(amplitude_callback=self._print_amplitude_debug)
        print("[orchestrator] Ready. Say the wake word, e.g. 'fish'. Press Ctrl+C to stop.")

    def _build_motion_service(self) -> MotionService:
        """
        Edit these pins/speeds to match your working motion_service.py test.
        These are the example pins from earlier.
        """
        motors = {
            "mouth": MotorConfig(
                in1=17,
                in2=27,
                en=22,
                forward_speed=82,
                reverse_speed=55,
                neutral_return_time=0.04,
            ),
            "tail": MotorConfig(
                in1=23,
                in2=24,
                en=25,
                forward_speed=72,
                reverse_speed=48,
                neutral_return_time=0.03,
            ),
            "body": MotorConfig(
                in1=5,
                in2=6,
                en=12,
                forward_speed=68,
                reverse_speed=45,
                neutral_return_time=0.03,
            ),
        }
        return MotionService(motors=motors)

    def _build_tool_registry(self, motion: MotionService) -> ToolRegistry:
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

        return registry

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        if not all([self.audio, self.stt, self.assistant, self.motion]):
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

        # Since TTS is skipped for now, use placeholder motion to represent speech.
        self._animate_response_without_tts(result.motion, result.speak)

    def _animate_response_without_tts(self, motion_hint: str, speak_text: str) -> None:
        assert self.motion is not None

        word_count = max(1, len(speak_text.split()))
        # Rough fake speech duration. Keep it bounded so bad model output does
        # not make the fish flap forever.
        duration_s = min(5.0, max(1.0, word_count * 0.18))

        if motion_hint in {"happy", "excited"}:
            self.motion.wiggle(cycles=1)
        elif motion_hint == "annoyed":
            self.motion.open_mouth()
        elif motion_hint == "thinking":
            self.motion.wiggle(cycles=1, tail=True, body=False)

        self.motion.speak_text_placeholder(duration_s=duration_s)

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
    orchestrator = FishseusDemoOrchestrator()

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
