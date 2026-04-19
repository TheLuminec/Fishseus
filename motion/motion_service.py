import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Optional

import RPi.GPIO as GPIO


@dataclass(frozen=True)
class MotorConfig:
    in1: int
    in2: int
    en: int
    forward_speed: float = 70.0
    reverse_speed: float = 55.0
    neutral_return_time: float = 0.08


class MotionService:
    """
    Non-blocking motion controller for a Billy Bass style fish.

    Design goals:
    - Orchestrator-facing methods return immediately.
    - A background worker owns motor timing and sequencing.
    - High-level motion helpers (open_mouth, wiggle, speak_audio, stop_all).
    - Safe shutdown and emergency stop support.

    Notes:
    - Assumes each motor is driven by an H-bridge with IN1/IN2 + EN PWM.
    - Assumes reverse pulses can be shorter because the mechanism tends to
      spring back toward neutral on its own.
    - Uses BCM GPIO numbering.
    """

    def __init__(
        self,
        motors: Dict[str, MotorConfig],
        pwm_frequency: int = 1000,
        body_wiggle_time: float = 0.18,
        tail_wiggle_time: float = 0.14,
        mouth_open_time: float = 0.09,
        mouth_close_time: float = 0.04,
        envelope_window_s: float = 0.03,
    ) -> None:
        self.motors = motors
        self.pwm_frequency = pwm_frequency
        self.body_wiggle_time = body_wiggle_time
        self.tail_wiggle_time = tail_wiggle_time
        self.mouth_open_time = mouth_open_time
        self.mouth_close_time = mouth_close_time
        self.envelope_window_s = envelope_window_s

        self._pwms: Dict[str, GPIO.PWM] = {}
        self._queue: Queue = Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_worker = threading.Event()
        self._cancel_motion = threading.Event()
        self._initialized = False
        self._lock = threading.Lock()

    # ---------------------------------------------------------------------
    # Public lifecycle
    # ---------------------------------------------------------------------
    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return

            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            for motor in self.motors.values():
                GPIO.setup(motor.in1, GPIO.OUT)
                GPIO.setup(motor.in2, GPIO.OUT)
                GPIO.setup(motor.en, GPIO.OUT)

            for name, motor in self.motors.items():
                pwm = GPIO.PWM(motor.en, self.pwm_frequency)
                pwm.start(0)
                self._pwms[name] = pwm
                self._set_idle_pins(name)

            self._stop_worker.clear()
            self._cancel_motion.clear()
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()
            self._initialized = True

    def shutdown(self) -> None:
        with self._lock:
            if not self._initialized:
                return

            self._stop_worker.set()
            self._cancel_motion.set()
            self._queue.put(("shutdown", {}))

            if self._thread is not None:
                self._thread.join(timeout=2.0)
                self._thread = None

            self._hard_stop_all()

            for name, pwm in list(self._pwms.items()):
                pwm.ChangeDutyCycle(0)
                pwm.stop()

            GPIO.cleanup()

            self._initialized = False

    # ---------------------------------------------------------------------
    # Public orchestrator-facing API
    # ---------------------------------------------------------------------
    def open_mouth(self, duration: Optional[float] = None, speed: Optional[float] = None) -> None:
        self._enqueue("open_mouth", {"duration": duration, "speed": speed})

    def wiggle(self, cycles: int = 2, tail: bool = True, body: bool = True) -> None:
        self._enqueue("wiggle", {"cycles": cycles, "tail": tail, "body": body})

    def speak_audio(self, wav_path: str | Path) -> None:
        self._enqueue("speak_audio", {"wav_path": str(wav_path)})

    def speak_text_placeholder(self, duration_s: float = 2.0) -> None:
        """
        Useful during early integration before TTS is wired in.
        Simulates speaking motion for a fixed time.
        """
        self._enqueue("speak_placeholder", {"duration_s": duration_s})

    def stop_all(self) -> None:
        """
        Cancel current motion and stop all motors immediately.
        """
        self._cancel_motion.set()
        self._hard_stop_all()

    # ---------------------------------------------------------------------
    # Worker + queue
    # ---------------------------------------------------------------------
    def _enqueue(self, command: str, payload: dict) -> None:
        if not self._initialized:
            raise RuntimeError("MotionService.initialize() must be called first")
        self._queue.put((command, payload))

    def _worker_loop(self) -> None:
        while not self._stop_worker.is_set():
            try:
                command, payload = self._queue.get(timeout=0.1)
            except Empty:
                continue

            if command == "shutdown":
                break

            self._cancel_motion.clear()

            try:
                if command == "open_mouth":
                    self._do_open_mouth(**payload)
                elif command == "wiggle":
                    self._do_wiggle(**payload)
                elif command == "speak_audio":
                    self._do_speak_audio(**payload)
                elif command == "speak_placeholder":
                    self._do_speak_placeholder(**payload)
            except Exception as exc:
                print(f"[MotionService] motion command failed: {exc}")
                self._hard_stop_all()
            finally:
                self._queue.task_done()

    # ---------------------------------------------------------------------
    # Motion implementations
    # ---------------------------------------------------------------------
    def _do_open_mouth(self, duration: Optional[float], speed: Optional[float]) -> None:
        if self._cancel_motion.is_set():
            return

        motor_name = "mouth"
        motor_cfg = self.motors[motor_name]
        open_time = duration if duration is not None else self.mouth_open_time
        open_speed = speed if speed is not None else motor_cfg.forward_speed

        self._drive_forward(motor_name, open_speed)
        self._sleep_with_cancel(open_time)
        self._soft_return_to_neutral(motor_name, self.mouth_close_time)

    def _do_wiggle(self, cycles: int, tail: bool, body: bool) -> None:
        for _ in range(max(1, cycles)):
            if self._cancel_motion.is_set():
                return

            if tail:
                self._pulse_motor("tail", self.tail_wiggle_time)

            if body:
                self._pulse_motor("body", self.body_wiggle_time)

    def _do_speak_placeholder(self, duration_s: float) -> None:
        start = time.monotonic()
        while time.monotonic() - start < duration_s:
            if self._cancel_motion.is_set():
                return
            self._do_open_mouth(duration=self.mouth_open_time, speed=None)
            self._sleep_with_cancel(0.06)

    def _do_speak_audio(self, wav_path: str) -> None:
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        with wave.open(str(path), "rb") as wav_file:
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            frame_rate = wav_file.getframerate()

            if sample_width not in (1, 2):
                raise ValueError("Only 8-bit or 16-bit PCM WAV files are supported")

            frames_per_window = max(1, int(frame_rate * self.envelope_window_s))

            while True:
                if self._cancel_motion.is_set():
                    return

                raw = wav_file.readframes(frames_per_window)
                if not raw:
                    break

                level = self._estimate_level(raw, sample_width, channels)
                self._mouth_from_level(level)

                # Keep the motion roughly aligned to the audio chunk timing.
                self._sleep_with_cancel(self.envelope_window_s)

        self._soft_return_to_neutral("mouth", self.mouth_close_time)

    # ---------------------------------------------------------------------
    # Low-level motion helpers
    # ---------------------------------------------------------------------
    def _pulse_motor(self, motor_name: str, forward_time: float) -> None:
        if self._cancel_motion.is_set():
            return

        cfg = self.motors[motor_name]
        self._drive_forward(motor_name, cfg.forward_speed)
        self._sleep_with_cancel(forward_time)
        self._soft_return_to_neutral(motor_name, cfg.neutral_return_time)

    def _mouth_from_level(self, level: float) -> None:
        """
        level is normalized 0.0 to 1.0.
        Uses a few thresholds instead of direct linear mapping so the fish
        looks more animated and less jittery.
        """
        if level < 0.08:
            self._soft_return_to_neutral("mouth", self.mouth_close_time)
            return

        cfg = self.motors["mouth"]

        if level < 0.18:
            speed = min(cfg.forward_speed, 45)
            duration = 0.03
        elif level < 0.35:
            speed = min(cfg.forward_speed, 60)
            duration = 0.05
        else:
            speed = cfg.forward_speed
            duration = 0.07

        self._drive_forward("mouth", speed)
        self._sleep_with_cancel(duration)
        self._soft_return_to_neutral("mouth", self.mouth_close_time)

    def _drive_forward(self, motor_name: str, speed: float) -> None:
        cfg = self.motors[motor_name]
        GPIO.output(cfg.in1, GPIO.HIGH)
        GPIO.output(cfg.in2, GPIO.LOW)
        self._pwms[motor_name].ChangeDutyCycle(self._clamp_pwm(speed))

    def _drive_reverse(self, motor_name: str, speed: float) -> None:
        cfg = self.motors[motor_name]
        GPIO.output(cfg.in1, GPIO.LOW)
        GPIO.output(cfg.in2, GPIO.HIGH)
        self._pwms[motor_name].ChangeDutyCycle(self._clamp_pwm(speed))

    def _soft_return_to_neutral(self, motor_name: str, reverse_time: float) -> None:
        if self._cancel_motion.is_set():
            self._set_idle_pins(motor_name)
            return

        cfg = self.motors[motor_name]
        if reverse_time > 0:
            self._drive_reverse(motor_name, cfg.reverse_speed)
            self._sleep_with_cancel(reverse_time)

        self._set_idle_pins(motor_name)
        self._pwms[motor_name].ChangeDutyCycle(0)

    def _set_idle_pins(self, motor_name: str) -> None:
        cfg = self.motors[motor_name]
        GPIO.output(cfg.in1, GPIO.LOW)
        GPIO.output(cfg.in2, GPIO.LOW)

    def _hard_stop_all(self) -> None:
        for name in self.motors:
            self._set_idle_pins(name)
            pwm = self._pwms.get(name)
            if pwm is not None:
                pwm.ChangeDutyCycle(0)

    def _sleep_with_cancel(self, duration_s: float) -> None:
        end = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < end:
            if self._cancel_motion.is_set() or self._stop_worker.is_set():
                return
            time.sleep(0.005)

    @staticmethod
    def _clamp_pwm(value: float) -> float:
        return max(0.0, min(100.0, float(value)))

    @staticmethod
    def _estimate_level(raw: bytes, sample_width: int, channels: int) -> float:
        if sample_width == 1:
            # 8-bit unsigned PCM
            samples = [abs(b - 128) / 128.0 for b in raw]
        else:
            # 16-bit signed PCM, little-endian
            samples = []
            for i in range(0, len(raw) - 1, 2):
                value = int.from_bytes(raw[i:i + 2], byteorder="little", signed=True)
                samples.append(abs(value) / 32768.0)

        if not samples:
            return 0.0

        if channels > 1:
            mono_samples = []
            for i in range(0, len(samples), channels):
                frame = samples[i:i + channels]
                if frame:
                    mono_samples.append(sum(frame) / len(frame))
            samples = mono_samples or samples

        return sum(samples) / len(samples)


if __name__ == "__main__":
    motors = {
        "mouth": MotorConfig(in1=17, in2=27, en=22, forward_speed=82, reverse_speed=55, neutral_return_time=0.04),
        "tail": MotorConfig(in1=23, in2=24, en=25, forward_speed=72, reverse_speed=48, neutral_return_time=0.03),
        "body": MotorConfig(in1=5, in2=6, en=12, forward_speed=68, reverse_speed=45, neutral_return_time=0.03),
    }

    motion = MotionService(motors=motors)
    motion.initialize()

    try:
        motion.wiggle(cycles=3)
        time.sleep(2.0)
        motion.open_mouth()
        time.sleep(1.0)
        motion.speak_text_placeholder(duration_s=3.0)
        time.sleep(4.0)
    finally:
        motion.shutdown()
