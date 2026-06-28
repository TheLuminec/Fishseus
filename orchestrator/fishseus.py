"""
fishseus.py — Production orchestrator for the Fishseus Billy Bass assistant.

Improvements over orchestrator_demo_tts.py:
- Integrated Flask web UI runs in a background daemon thread (zero extra
  resource cost when nobody is accessing it).
- Microphone capture is stopped while the fish is speaking, then restarted
  with a short cooldown, so the fish cannot trigger itself.
- Piper TTS runs as a persistent subprocess — the ONNX model is loaded once
  at start-up and stays in memory, eliminating the per-utterance reload delay.
- Robust error recovery: exceptions in STT, LLM, or TTS are caught
  individually; the fish wiggles or speaks a fallback line and keeps
  listening rather than crashing.

Flow:
    USB mic
      -> audio_service.record_until_silence()
      -> stt_service.transcribe_file()         (whisper.cpp)
      -> wake-word check
      -> [mic capture paused]
      -> assistant_service.handle_user_text()  (LLM)
      -> tts_service.synthesize()              (Piper daemon — fast)
      -> motion_service.speak_audio() + tts_service.play_wav()  (synced)
      -> cooldown sleep
      -> [mic capture resumed]

Configuration is read from config/fish_config.json.  No code changes needed
for tuning hardware or model settings.

Usage:
    python orchestrator/fishseus.py [--port 8000] [--no-web]
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------
# Path setup — mirrors the pattern used by the demo orchestrators
# -----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for _subdir in ("audio", "stt", "llm", "assistant", "motion", "tts", "web"):
    _candidate = PROJECT_ROOT / _subdir
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from audio_service import AudioConfig, AudioService
from stt_service import SttConfig, SttService
from llm_service import LlmConfig, LlmService
from assistant_service import AssistantConfig, AssistantService, Tool, ToolRegistry
from motion_service import MotionService, MotorConfig
from tts_service import TtsConfig, TtsService


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(tag: str, msg: str) -> None:
    print(f"[{_ts()}] [{tag}] {msg}", flush=True)


# -----------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------

class Fishseus:
    """
    Production orchestrator.  Owns all service instances and the main loop.
    """

    # seconds of silence after TTS playback before re-enabling the mic
    POST_SPEECH_COOLDOWN = 1.5

    def __init__(self, web_port: int = 8000, enable_web: bool = True) -> None:
        self.web_port = web_port
        self.enable_web = enable_web

        self.running = False
        self._speaking = False      # True while TTS is playing (mic suppressed)
        self._speaking_lock = threading.Lock()

        # Services — populated in initialize()
        self.audio: Optional[AudioService] = None
        self.stt: Optional[SttService] = None
        self.llm: Optional[LlmService] = None
        self.assistant: Optional[AssistantService] = None
        self.motion: Optional[MotionService] = None
        self.tts: Optional[TtsService] = None

        self._speech_wav = PROJECT_ROOT / "tmp" / "latest_speech.wav"

    # -------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------

    def initialize(self) -> None:
        config_file = PROJECT_ROOT / "config" / "fish_config.json"
        app_config: dict = {}
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as fh:
                app_config = json.load(fh)

        config_dir = config_file.parent

        audio_cfg   = app_config.get("audio", {})
        stt_cfg     = app_config.get("stt", {})
        llm_cfg     = app_config.get("llm", {})
        tts_cfg     = app_config.get("tts", {})
        motion_cfg  = app_config.get("motion", {})
        asst_cfg    = app_config.get("assistant", {})

        self._speech_wav.parent.mkdir(parents=True, exist_ok=True)

        pers_path = (config_dir / asst_cfg.get(
            "personality_path", "../config/personality_prompt.txt"
        )).resolve()
        mem_path  = PROJECT_ROOT / "data" / "assistant_memory.json"
        hist_path = PROJECT_ROOT / "data" / "conversation_log.jsonl"
        tts_out   = (config_dir / tts_cfg.get("output_dir", "../tmp/tts")).resolve()
        tts_out.mkdir(parents=True, exist_ok=True)
        mem_path.parent.mkdir(parents=True, exist_ok=True)

        # Audio
        _log("init", "Starting audio service…")
        self.audio = AudioService(AudioConfig(
            device              = audio_cfg.get("device", "plughw:3,0"),
            sample_rate         = int(audio_cfg.get("sample_rate", 16000)),
            channels            = int(audio_cfg.get("channels", 1)),
            speech_threshold    = float(audio_cfg.get("speech_threshold", 0.003)),
            silence_threshold   = float(audio_cfg.get("silence_threshold", 0.0008)),
            silence_timeout_s   = float(audio_cfg.get("silence_timeout_s", 1.0)),
            max_record_seconds  = float(audio_cfg.get("max_record_seconds", 12.0)),
            pre_roll_seconds    = float(audio_cfg.get("pre_roll_seconds", 0.4)),
        ))
        self.audio.initialize()

        # STT
        _log("init", "Starting STT service…")
        self.stt = SttService(SttConfig(
            whisper_binary = str((config_dir / stt_cfg.get(
                "whisper_binary", "../stt/whisper.cpp/build/bin/whisper-cli"
            )).resolve()),
            model_path = str((config_dir / stt_cfg.get(
                "model_path", "../stt/whisper.cpp/models/ggml-base.en.bin"
            )).resolve()),
            threads         = int(stt_cfg.get("threads", 4)),
            wake_words      = stt_cfg.get("wake_words", ["fish", "fishseus", "hey fish"]),
            strip_wake_word = bool(stt_cfg.get("strip_wake_word", True)),
        ))
        self.stt.initialize()

        # LLM
        _log("init", "Starting LLM service…")
        self.llm = LlmService(LlmConfig(
            endpoint_url      = llm_cfg.get("endpoint_url", "http://ollama.angelfish-gamma.ts.net/v1/chat/completions"),
            model             = llm_cfg.get("model", "qwen2.5:3b"),
            timeout_s         = 45.0,
            retries           = 1,
            temperature       = float(llm_cfg.get("temperature", 0.7)),
            max_tokens        = int(llm_cfg.get("max_tokens", 512)),
            disable_reasoning = bool(llm_cfg.get("disable_reasoning", True)),
        ))

        # TTS  — persistent=True keeps piper loaded in memory
        _log("init", "Starting TTS service (persistent piper daemon)…")
        self.tts = TtsService(TtsConfig(
            piper_binary  = str((config_dir / tts_cfg.get("piper_binary", "../tts/.venv/bin/piper")).resolve()),
            voices_dir    = Path((config_dir / tts_cfg.get("voices_dir", "../tts/voices")).resolve()),
            default_voice = tts_cfg.get("default_voice", "en_US-arctic-medium"),
            audio_device  = tts_cfg.get("audio_device", "plughw:0,0"),
            output_dir    = tts_out,
            persistent    = True,
        ))
        self.tts.initialize()
        _log("init", f"TTS voices available: {self.tts.available_voices()}")

        # Motion
        _log("init", "Starting motion service…")
        self.motion = self._build_motion_service(motion_cfg)
        self.motion.initialize()

        # Assistant
        _log("init", "Starting assistant service…")
        tool_registry = self._build_tool_registry()
        self.assistant = AssistantService(
            llm=self.llm,
            config=AssistantConfig(
                assistant_name = asst_cfg.get("assistant_name", "Fishseus"),
                user_name      = asst_cfg.get("user_name", "User"),
                personality_path = pers_path,
                memory_path    = mem_path,
                history_path   = hist_path,
            ),
            tool_registry=tool_registry,
        )

        # Web UI
        if self.enable_web:
            self._start_web_server()

        # Begin audio capture
        _log("init", "Starting audio capture…")
        self.audio.start_capture(amplitude_callback=self._on_amplitude)
        _log("init", "Ready. Say 'fish' or 'hey fish' to wake me up. Ctrl-C to stop.")

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def run(self) -> None:
        if self.audio is None:
            raise RuntimeError("initialize() must be called before run()")

        self.running = True

        while self.running:
            # While speaking, drain any queued audio and skip processing.
            with self._speaking_lock:
                is_speaking = self._speaking
            if is_speaking:
                time.sleep(0.05)
                continue

            _log("listen", "Waiting for speech…")
            try:
                wav_path = self.audio.record_until_silence(self._speech_wav)
            except Exception as exc:
                _log("audio", f"record_until_silence error: {exc}")
                time.sleep(0.5)
                continue

            if wav_path is None:
                continue  # silence timeout — loop back

            _log("stt", f"Transcribing {wav_path}…")
            try:
                stt_result = self.stt.transcribe_file(wav_path)
            except Exception as exc:
                _log("stt", f"Transcription failed: {exc}")
                self.motion.wiggle(cycles=1)
                continue

            transcript = stt_result.text.strip()
            _log("stt", f"text='{transcript}'  wake={stt_result.wake_detected}")

            if not stt_result.wake_detected:
                continue

            command = stt_result.command_text.strip() or "Hello."
            self._handle_command(command)

    # -------------------------------------------------------------------
    # Command handling
    # -------------------------------------------------------------------

    def _handle_command(self, command: str) -> None:
        assert self.assistant is not None
        assert self.motion is not None

        _log("assistant", f"Command: '{command}'")

        # Immediate acknowledgement wiggle while LLM thinks.
        self.motion.wiggle(cycles=1)

        try:
            result = self.assistant.handle_user_text(command)
        except Exception as exc:
            _log("assistant", f"AssistantService error: {exc}")
            self._speak_with_motion("Sorry, my brain got tangled in seaweed.", "annoyed")
            return

        _log("assistant", f"speak='{result.speak}'  motion={result.motion}  tools={[c.name for c in result.tool_calls]}  elapsed={result.elapsed_s:.2f}s")

        self._speak_with_motion(result.speak, result.motion)

    # -------------------------------------------------------------------
    # Speaking (mic suppression + TTS + motion)
    # -------------------------------------------------------------------

    def _speak_with_motion(self, text: str, motion_hint: str) -> None:
        assert self.tts is not None
        assert self.motion is not None
        assert self.audio is not None

        # Suppress mic so the fish can't hear itself.
        with self._speaking_lock:
            self._speaking = True
        self.audio.stop_capture()

        try:
            _log("tts", "Synthesising response…")
            try:
                tts_wav = self.tts.synthesize(text)
            except Exception as exc:
                _log("tts", f"Synthesis failed: {exc}")
                self.motion.speak_text_placeholder(duration_s=1.5)
                return

            # Pre-speech reaction based on emotion hint.
            self._pre_speech_reaction(motion_hint)

            # Start mouth animation and audio playback at the same time.
            # speak_audio() is non-blocking (queues work to the motion thread).
            # play_wav() blocks until the audio finishes.
            _log("tts", f"Playing {tts_wav}…")
            self.motion.speak_audio(tts_wav)
            try:
                self.tts.play_wav(tts_wav, blocking=True)
            except Exception as exc:
                _log("tts", f"Playback failed: {exc}")

            # Wait for room echo to die down before re-enabling the mic.
            time.sleep(self.POST_SPEECH_COOLDOWN)

        finally:
            # Always re-enable the mic, even if TTS or playback crashed.
            self.audio.start_capture(amplitude_callback=self._on_amplitude)
            with self._speaking_lock:
                self._speaking = False

    def _pre_speech_reaction(self, motion_hint: str) -> None:
        if motion_hint in {"happy", "excited"}:
            self.motion.wiggle(cycles=1)
        elif motion_hint == "thinking":
            self.motion.wiggle(cycles=1, tail=True, body=False)
        elif motion_hint == "annoyed":
            self.motion.open_mouth()

    # -------------------------------------------------------------------
    # Amplitude callback (feeds web UI graph)
    # -------------------------------------------------------------------

    def _on_amplitude(self, amp: float) -> None:
        # Forward to web server's rolling buffer so the UI graph stays live.
        try:
            import web_server as _web
            _web._amplitude_callback(amp)
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Service construction helpers
    # -------------------------------------------------------------------

    def _build_motion_service(self, motion_cfg: dict) -> MotionService:
        motors_raw = motion_cfg.get("motors", {})
        motors = {}
        for name, m in motors_raw.items():
            motors[name] = MotorConfig(
                in1                = int(m.get("in1", 0)),
                in2                = int(m.get("in2", 0)),
                en                 = int(m.get("en", 0)),
                forward_speed      = float(m.get("forward_speed", 70)),
                reverse_speed      = float(m.get("reverse_speed", 55)),
                neutral_return_time= float(m.get("neutral_return_time", 0.08)),
            )

        if not motors:
            motors = {
                "mouth": MotorConfig(17, 27, 22, 82, 55, 0.04),
                "tail":  MotorConfig(23, 24, 25, 72, 48, 0.03),
                "body":  MotorConfig(5,  6,  12, 68, 45, 0.03),
            }

        return MotionService(
            motors           = motors,
            pwm_frequency    = int(motion_cfg.get("pwm_frequency", 1000)),
            body_wiggle_time = float(motion_cfg.get("body_wiggle_time", 0.18)),
            tail_wiggle_time = float(motion_cfg.get("tail_wiggle_time", 0.14)),
            mouth_open_time  = float(motion_cfg.get("mouth_open_time", 0.09)),
            mouth_close_time = float(motion_cfg.get("mouth_close_time", 0.04)),
            envelope_window_s= float(motion_cfg.get("envelope_window_s", 0.18)),
        )

    def _build_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()

        def wiggle(cycles: int = 1) -> str:
            cycles = max(1, min(int(cycles), 5))
            self.motion.wiggle(cycles=cycles)
            return f"wiggle queued for {cycles} cycle(s)"

        def open_mouth() -> str:
            self.motion.open_mouth()
            return "mouth open queued"

        def set_mode(mode: str) -> str:
            allowed = {"assistant", "bluetooth"}
            if mode not in allowed:
                raise ValueError(f"mode must be one of {sorted(allowed)}")
            return f"mode switch requested: {mode}"

        def get_current_time() -> str:
            return time.strftime("%Y-%m-%d %H:%M:%S")

        def list_voices() -> str:
            voices = self.tts.available_voices()
            return ", ".join(voices) if voices else "no voices found"

        def set_voice(voice: str) -> str:
            self.tts.set_voice(voice)
            return f"voice set to {voice}"

        tools = [
            Tool("wiggle",           "Make the fish wiggle. Args: cycles int 1-5.", wiggle,       "safe"),
            Tool("open_mouth",       "Open the fish mouth once. No args.",           open_mouth,   "safe"),
            Tool("set_mode",         "Request mode switch. Args: mode 'assistant' or 'bluetooth'.", set_mode, "safe"),
            Tool("get_current_time", "Get the current system time. No args.",        get_current_time, "safe"),
            Tool("list_voices",      "List available Piper TTS voices. No args.",    list_voices,  "safe"),
            Tool("set_voice",        "Set the TTS voice. Args: voice name string.",  set_voice,    "safe"),
        ]
        for t in tools:
            registry.register(t)
        return registry

    # -------------------------------------------------------------------
    # Web server
    # -------------------------------------------------------------------

    def _start_web_server(self) -> None:
        try:
            import web_server as _web
            _web.set_services(
                audio     = self.audio,
                llm       = self.llm,
                assistant = self.assistant,
                motion    = self.motion,
                tts       = self.tts,
            )
            thread = threading.Thread(
                target=_web.app.run,
                kwargs={
                    "host":        "0.0.0.0",
                    "port":        self.web_port,
                    "debug":       False,
                    "threaded":    True,
                    "use_reloader": False,
                },
                name="WebServer",
                daemon=True,
            )
            thread.start()
            _log("web", f"Control panel at http://0.0.0.0:{self.web_port}/")
        except Exception as exc:
            _log("web", f"Web server failed to start: {exc}")

    # -------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------

    def stop(self) -> None:
        self.running = False

    def shutdown(self) -> None:
        _log("shutdown", "Shutting down…")

        if self.audio is not None:
            try:
                self.audio.shutdown()
            except Exception as exc:
                _log("shutdown", f"Audio error: {exc}")

        if self.stt is not None:
            try:
                self.stt.shutdown()
            except Exception as exc:
                _log("shutdown", f"STT error: {exc}")

        if self.tts is not None:
            try:
                self.tts.shutdown()
            except Exception as exc:
                _log("shutdown", f"TTS error: {exc}")

        if self.motion is not None:
            try:
                self.motion.stop_all()
                self.motion.shutdown()
            except Exception as exc:
                _log("shutdown", f"Motion error: {exc}")

        if self.llm is not None:
            try:
                self.llm.close()
            except Exception as exc:
                _log("shutdown", f"LLM error: {exc}")

        _log("shutdown", "Done.")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Fishseus production orchestrator")
    parser.add_argument("--port",    type=int, default=8000, help="Web UI port (default 8000)")
    parser.add_argument("--no-web",  action="store_true",    help="Disable the Flask web UI")
    args = parser.parse_args()

    orch = Fishseus(web_port=args.port, enable_web=not args.no_web)

    def _handle_signal(signum, frame):
        _log("signal", f"Received signal {signum}, stopping…")
        orch.stop()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        orch.initialize()
        orch.run()
    except KeyboardInterrupt:
        orch.stop()
    finally:
        orch.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
