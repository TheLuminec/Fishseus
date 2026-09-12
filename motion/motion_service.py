"""
motion_service.py

Thin motion-control service for a Billy Bass style fish (Pi-only — talks to
the Fishseus RPi HAT over I2C).

The Pi no longer drives motor pins itself. The HAT carries an ATtiny1614
motor coprocessor that owns a PCA9685 and two TB6612FNG H-bridges:

    Pi  --I2C(master)-->  ATtiny1614  --I2C-->  PCA9685 --> 2x TB6612 --> M1..M4
       GPIO2/GPIO3, addr 0x28

So this service is now a protocol client. Motors are addressed by channel
(0..3 = M1..M4 on the silkscreen) rather than by IN1/IN2/EN GPIO numbers, and
the ATtiny's non-blocking step sequencer executes the actual pulse timing —
which means a pulse is millisecond-accurate even while Python is busy.

Responsibilities:
- Arm/disarm the HAT's hardware kill line and drive the motors over I2C
- Offer high-level motion helpers (open_mouth, wiggle, speak_audio, stop_all)
- Animate the mouth from a WAV's amplitude envelope while TTS plays
- Surface the coprocessor's telemetry (firmware, temperature, faults)
- Support safe shutdown and emergency stop

Non-responsibilities:
- No audio capture, STT, TTS, assistant, or LLM logic

Notes:
- Nothing moves until the HAT is armed. /OE has a pull-up, so at reset or
  brown-out every output is forced low and both bridges sit in standby.
  initialize() arms the board (auto_enable) and shutdown() disarms it.
- Speeds are percent (0-100) at this API's boundary, matching the old GPIO
  duty cycles and the web UI's sliders. They are scaled to the firmware's
  -255..255 on the wire.
- Reverse pulses can be shorter because the mechanism springs back to neutral.
- Imports smbus2 at module load and opens /dev/i2c-N in initialize(), so this
  module is only useful on the Pi.

Example:
    motion = MotionService(MotionConfig())   # built-in default channel map
    motion.initialize()
    motion.wiggle(cycles=3)
    motion.speak_audio("tmp/tts/reply.wav")
    motion.shutdown()

Orchestrator usage:
    motion = MotionService(MotionConfig(motors={...}, i2c_bus=1, ...))
    motion.initialize()
    ...
    motion.stop_all()
    motion.shutdown()
"""

import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterable, List, Optional, Tuple

from smbus2 import SMBus, i2c_msg

from services import Service, ServiceConfig, ServiceError


# ---------------------------------------------------------------------------
# ATtiny1614 bridge protocol (see attiny1614_bridge/pi_link.h)
# ---------------------------------------------------------------------------

BRIDGE_I2C_ADDR = 0x28

CMD_SET       = 0x10   # m, sp_hi, sp_lo            int16 -255..255
CMD_STOP_ALL  = 0x11   # —                          coast every motor
CMD_ENABLE    = 0x12   # en                         /OE: 1 = outputs live
CMD_BRAKE     = 0x13   # m
CMD_COAST     = 0x14   # m
CMD_SEQ_CLEAR = 0x15   # m
CMD_SEQ_PUSH  = 0x16   # m, from_hi, from_lo, to_hi, to_lo, ms_hi, ms_lo, end
CMD_SEQ_START = 0x17   # m, loops                   loops 0 = forever
CMD_SEQ_STOP  = 0x18   # m
CMD_STANDBY   = 0x19   # en                         1 = drivers asleep
CMD_SET_FREQ  = 0x1A   # hz_hi, hz_lo

REG_STATUS    = 0x30   # 8 bytes of telemetry
REG_SPEEDS    = 0x31   # four signed big-endian int16, M1..M4

# Sequencer step end actions.
END_HOLD  = 0
END_BRAKE = 1
END_COAST = 2

MOTOR_COUNT         = 4      # M1..M4
SPEED_COUNTS_MAX    = 255    # firmware speed range is -255..255
SEQ_STEPS_PER_MOTOR = 6      # firmware step buffer, per motor
PWM_FREQ_MIN_HZ     = 24     # PCA9685 prescale limits
PWM_FREQ_MAX_HZ     = 1526
STEP_MS_MAX         = 65535  # durationMs is a uint16


class MotionServiceError(ServiceError):
    pass


@dataclass(frozen=True)
class MotorConfig:
    """
    One motor on the HAT.

    channel is 0-based (0..3 = M1..M4, connectors U13/U14/U15/U16). Speeds are
    percent (0-100) and are scaled to the firmware's -255..255 counts.

    invert flips this motor's sign convention in software. The firmware's
    convention is fixed to the TB6612 truth table (positive = IN1 high =
    AO1/BO1 driven), so use this when a motor is wired backwards and you would
    rather not swap the two wires in its connector.
    """
    channel: int
    forward_speed: float = 70.0
    reverse_speed: float = 55.0
    neutral_return_time: float = 0.08
    invert: bool = False


def _default_motors() -> Dict[str, MotorConfig]:
    return {
        "mouth": MotorConfig(0, 82, 55, 0.04),
        "tail":  MotorConfig(1, 72, 48, 0.03),
        "body":  MotorConfig(2, 68, 45, 0.03),
    }


@dataclass(frozen=True)
class MotionConfig(ServiceConfig):
    module_name: str = "motion"

    motors: Dict[str, MotorConfig] = field(default_factory=_default_motors)

    # --- link to the HAT ---
    i2c_bus: int = 1                    # /dev/i2c-1 = GPIO2/GPIO3
    i2c_address: int = BRIDGE_I2C_ADDR
    i2c_retries: int = 1
    command_gap_s: float = 0.003        # the ATtiny's command queue is 4 deep
    auto_enable: bool = True            # arm the kill line during initialize()
    heartbeat_interval_s: float = 1.0   # idle status polls; 0 disables

    # --- motion feel ---
    pwm_frequency: int = 1500           # PCA9685 tops out at 1526 Hz
    body_wiggle_time: float = 0.18
    tail_wiggle_time: float = 0.14
    mouth_open_time: float = 0.09
    mouth_close_time: float = 0.04
    envelope_window_s: float = 0.18

    def validate(self) -> bool:
        if not self.motors:
            raise MotionServiceError("No motors configured")

        seen: Dict[int, str] = {}
        for name, motor in self.motors.items():
            if not 0 <= motor.channel < MOTOR_COUNT:
                raise MotionServiceError(
                    f"Motor '{name}' channel {motor.channel} is out of range "
                    f"(0..{MOTOR_COUNT - 1} = M1..M{MOTOR_COUNT})"
                )
            if motor.channel in seen:
                raise MotionServiceError(
                    f"Motors '{seen[motor.channel]}' and '{name}' both claim "
                    f"channel {motor.channel}"
                )
            seen[motor.channel] = name

        if not PWM_FREQ_MIN_HZ <= self.pwm_frequency <= PWM_FREQ_MAX_HZ:
            raise MotionServiceError(
                f"pwm_frequency must be {PWM_FREQ_MIN_HZ}..{PWM_FREQ_MAX_HZ} Hz, "
                f"got {self.pwm_frequency}"
            )
        return True


class MotionService(Service):
    """
    Non-blocking motion controller for a Billy Bass style fish.

    Design goals:
    - Orchestrator-facing methods return immediately.
    - A background worker owns motion sequencing; the HAT owns pulse timing.
    - High-level motion helpers (open_mouth, wiggle, speak_audio, stop_all).
    - Safe shutdown and emergency stop support.

    Notes:
    - Every short move is pushed to the ATtiny's step sequencer as a forward
      step plus a reverse step toward neutral, then started. The worker sleeps
      for the same span so queued motions stay in order.
    - stop_all() and emergency_stop() are single writes and do not depend on
      the worker being responsive.
    """

    def __init__(self, config: MotionConfig = MotionConfig()) -> None:
        self.config = config
        # Mirror config onto attributes the motion body uses.
        self.motors = config.motors
        self.i2c_bus = config.i2c_bus
        self.address = config.i2c_address
        self.i2c_retries = config.i2c_retries
        self.command_gap_s = config.command_gap_s
        self.pwm_frequency = config.pwm_frequency
        self.body_wiggle_time = config.body_wiggle_time
        self.tail_wiggle_time = config.tail_wiggle_time
        self.mouth_open_time = config.mouth_open_time
        self.mouth_close_time = config.mouth_close_time
        self.envelope_window_s = config.envelope_window_s

        self._bus: Optional[SMBus] = None
        self._bus_lock = threading.Lock()
        self._bus_errors = 0
        self._armed = False

        self._telemetry: Optional[dict] = None
        self._telemetry_at = 0.0

        # Motors known to be parked, so a silent audio window does not
        # re-issue a close on every pass.
        self._at_neutral: Dict[str, bool] = {name: False for name in self.motors}

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
        self.config.validate()
        with self._lock:
            if self._initialized:
                return

            try:
                self._bus = SMBus(self.i2c_bus)
            except Exception as exc:
                raise MotionServiceError(
                    f"Cannot open I2C bus {self.i2c_bus}: {exc}. Enable I2C "
                    f"(raspi-config) and check the HAT is seated."
                ) from None

            telemetry = self._read_status(force=True)
            if telemetry is None:
                self._close_bus()
                raise MotionServiceError(
                    f"No response from the motor bridge at 0x{self.address:02X} "
                    f"on bus {self.i2c_bus}. Check the ATtiny1614 is flashed "
                    f"and powered."
                )
            if not telemetry["pca_ok"]:
                print("[MotionService] bridge reports PCA9685 errors — check "
                      "the sub bus (SUB SDA/SCL)")

            self._set_pwm_frequency_locked(self.pwm_frequency)
            self._hard_stop_all()

            if self.config.auto_enable:
                self._arm_locked(True)

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

            # Coast everything, then drop the kill line so the board is inert
            # whether or not this process comes back.
            try:
                self._hard_stop_all()
                self._arm_locked(False)
            except MotionServiceError as exc:
                print(f"[MotionService] disarm failed: {exc}")

            self._close_bus()
            self._initialized = False

    def reset(self) -> bool:
        self.shutdown()
        self.initialize()
        return True

    def status(self) -> dict:
        telemetry = self._read_status() if self._initialized else None

        state = "uninitialized"
        if self._initialized:
            state = "ok" if telemetry is not None else "bridge_unreachable"

        return {
            "enabled": self.enabled,
            "service": state,
            "worker_alive": self._thread.is_alive() if self._thread else False,
            "motors": list(self.motors.keys()),
            "queue_size": self._queue.qsize(),
            "armed": self._armed,
            "i2c_bus": self.i2c_bus,
            "i2c_address": f"0x{self.address:02X}",
            "bus_errors": self._bus_errors,
            "bridge": telemetry,
        }

    # ---------------------------------------------------------------------
    # Public orchestrator-facing API
    # ---------------------------------------------------------------------
    def open_mouth(self, duration: Optional[float] = None, speed: Optional[float] = None) -> None:
        self._enqueue("open_mouth", {"duration": duration, "speed": speed})

    def wiggle(self, cycles: int = 2, tail: bool = True, body: bool = True, speed_scale: float = 1.0) -> None:
        self._enqueue("wiggle", {"cycles": cycles, "tail": tail, "body": body, "speed_scale": speed_scale})

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
        Cancel current motion and coast all motors immediately.
        """
        self._cancel_motion.set()
        self._hard_stop_all()

    def emergency_stop(self) -> None:
        """
        Drop the hardware kill line.

        This is a single write to /OE on the coprocessor: every PCA9685 output
        goes low, which also drops the shared TB6612 STBY, so both bridges go
        high-Z. It does not depend on the motion queue being healthy. Call
        arm() to bring the board back.
        """
        self._cancel_motion.set()
        self._arm(False)

    def arm(self, enabled: bool = True) -> None:
        """
        Raise (or drop) the HAT's output-enable line. Motors cannot move while
        disarmed, which is the state the board powers up in.
        """
        self._arm(enabled)

    def set_pwm_frequency(self, hz: int) -> None:
        """Retune the PCA9685 carrier (24..1526 Hz) on the live board."""
        hz = int(hz)
        if not PWM_FREQ_MIN_HZ <= hz <= PWM_FREQ_MAX_HZ:
            raise ValueError(
                f"pwm_frequency must be {PWM_FREQ_MIN_HZ}..{PWM_FREQ_MAX_HZ} Hz"
            )
        self._require_init()
        self._set_pwm_frequency_locked(hz)
        self.pwm_frequency = hz

    def motor_speeds(self) -> Dict[str, float]:
        """
        Read back what the coprocessor is actually driving, as percent
        (-100..100), keyed by configured motor name.
        """
        self._require_init()
        raw = self._read_register(REG_SPEEDS)
        if raw is None or len(raw) < 8:
            return {}

        speeds: Dict[str, float] = {}
        for name, cfg in self.motors.items():
            counts = _int16_be(raw, cfg.channel * 2)
            if cfg.invert:
                counts = -counts
            speeds[name] = round(counts * 100.0 / SPEED_COUNTS_MAX, 1)
        return speeds

    # -----------------------------------------------------------------
    # Direct (non-queued) motor control for fine tuning
    # -----------------------------------------------------------------
    def direct_drive(self, motor_name: str, direction: str, speed: float) -> None:
        """
        Immediately drive a motor at the given speed and direction.

        This bypasses the worker queue entirely so the web UI can implement
        hold-to-run buttons for real-time motor tuning. The firmware cancels
        any sequence running on that motor when it takes a direct speed, so
        this always wins.

        Parameters:
            motor_name: One of the configured motor names (mouth, tail, body).
            direction:  "forward" or "reverse".
            speed:      Percent 0-100.
        """
        self._require_init()
        self._motor(motor_name)   # raises on an unknown name

        if direction == "forward":
            sign = 1
        elif direction == "reverse":
            sign = -1
        else:
            raise ValueError(f"direction must be 'forward' or 'reverse', got '{direction}'")

        self._set_speed(motor_name, sign * _percent_to_counts(speed))

    def direct_stop(self, motor_name: str | None = None) -> None:
        """
        Immediately coast a specific motor (or all motors if name is None).

        Does not cancel queued motions — use stop_all() for that.
        """
        self._require_init()
        if motor_name is None:
            self._hard_stop_all()
            return

        cfg = self._motor(motor_name)
        self._write(CMD_COAST, [cfg.channel])
        self._at_neutral[motor_name] = True

    # ---------------------------------------------------------------------
    # Worker + queue
    # ---------------------------------------------------------------------
    def _enqueue(self, command: str, payload: dict) -> None:
        self._require_init()
        self._queue.put((command, payload))

    def _worker_loop(self) -> None:
        heartbeat = self.config.heartbeat_interval_s

        while not self._stop_worker.is_set():
            try:
                command, payload = self._queue.get(timeout=0.1)
            except Empty:
                # Idle: poll the bridge. Any read also refreshes the firmware's
                # "the Pi is alive" timestamp, which is what LINK_FAILSAFE_MS
                # watches if you turn it on.
                if heartbeat > 0 and time.monotonic() - self._telemetry_at >= heartbeat:
                    self._read_status()
                continue

            if command == "shutdown":
                self._queue.task_done()
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
                try:
                    self._hard_stop_all()
                except MotionServiceError:
                    pass
            finally:
                self._queue.task_done()

    # ---------------------------------------------------------------------
    # Motion implementations
    # ---------------------------------------------------------------------
    def _do_open_mouth(self, duration: Optional[float], speed: Optional[float]) -> None:
        if self._cancel_motion.is_set():
            return

        cfg = self.motors["mouth"]
        open_time = duration if duration is not None else self.mouth_open_time
        open_speed = speed if speed is not None else cfg.forward_speed

        self._pulse("mouth", open_time, self.mouth_close_time, open_speed)

    def _do_wiggle(self, cycles: int, tail: bool, body: bool, speed_scale: float = 1.0) -> None:
        for _ in range(max(1, cycles)):
            if self._cancel_motion.is_set():
                return

            if tail and "tail" in self.motors:
                self._pulse_motor("tail", self.tail_wiggle_time, speed_scale)

            if body and "body" in self.motors:
                self._pulse_motor("body", self.body_wiggle_time, speed_scale)

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

                # Budget the whole window: the mouth pulse runs inside it and
                # only the remainder is slept, so the motion does not drift
                # behind the audio it is supposed to be tracking.
                window_start = time.monotonic()
                level = self._estimate_level(raw, sample_width, channels)
                self._mouth_from_level(level)
                self._sleep_with_cancel(
                    self.envelope_window_s - (time.monotonic() - window_start)
                )

        self._return_to_neutral("mouth", self.mouth_close_time)

    # ---------------------------------------------------------------------
    # Low-level motion helpers
    # ---------------------------------------------------------------------
    def _pulse_motor(self, motor_name: str, forward_time: float, speed_scale: float = 1.0) -> None:
        cfg = self.motors[motor_name]
        speed = max(0.0, min(100.0, cfg.forward_speed * speed_scale))
        self._pulse(motor_name, forward_time, cfg.neutral_return_time, speed)

    def _pulse(self, motor_name: str, forward_time: float, reverse_time: float,
               forward_speed: float) -> None:
        """
        One move: drive forward, then a short reverse toward neutral, then coast.

        Both steps are handed to the coprocessor's sequencer as a single
        program, so the timing is the ATtiny's millis() rather than Python's
        scheduler. The worker then sleeps for the same span, which keeps
        queued motions from overlapping each other.
        """
        if self._cancel_motion.is_set():
            return

        cfg = self.motors[motor_name]
        forward_ms = _to_ms(forward_time)
        reverse_ms = _to_ms(reverse_time)

        steps: List[Tuple[int, int, int]] = []
        if forward_ms:
            steps.append((_percent_to_counts(forward_speed), forward_ms, END_HOLD))
        if reverse_ms:
            steps.append((-_percent_to_counts(cfg.reverse_speed), reverse_ms, END_HOLD))
        if not steps:
            return

        # Whatever the last step is, the motor ends up high-Z rather than held.
        speed, ms, _ = steps[-1]
        steps[-1] = (speed, ms, END_COAST)

        self._run_sequence(cfg, steps)
        self._at_neutral[motor_name] = False

        self._sleep_with_cancel((forward_ms + reverse_ms) / 1000.0)

        if self._cancel_motion.is_set():
            # Do not leave the ATtiny running a program we walked away from.
            self._write(CMD_COAST, [cfg.channel])
        self._at_neutral[motor_name] = True

    def _return_to_neutral(self, motor_name: str, reverse_time: float) -> None:
        if self._at_neutral.get(motor_name):
            return
        if self._cancel_motion.is_set():
            self._write(CMD_COAST, [self.motors[motor_name].channel])
            self._at_neutral[motor_name] = True
            return
        self._pulse(motor_name, 0.0, reverse_time, 0.0)

    def _mouth_from_level(self, level: float) -> None:
        """
        level is normalized 0.0 to 1.0.
        Uses a few thresholds instead of direct linear mapping so the fish
        looks more animated and less jittery.
        """
        if level < 0.08:
            self._return_to_neutral("mouth", self.mouth_close_time)
            return

        cfg = self.motors["mouth"]

        if level < 0.18:
            speed = min(cfg.forward_speed, 45)
            duration = 0.06
        elif level < 0.35:
            speed = min(cfg.forward_speed, 60)
            duration = 0.09
        else:
            speed = cfg.forward_speed
            duration = 0.12

        self._pulse("mouth", duration, self.mouth_close_time, speed)

    def _run_sequence(self, cfg: MotorConfig, steps: Iterable[Tuple[int, int, int]]) -> None:
        """
        Load and start a motor program: clear, push each step, run once.

        Steps are (speed_counts, duration_ms, end_action) with the speed in the
        motor's own sign convention; invert is applied here. The firmware holds
        SEQ_STEPS_PER_MOTOR of them and its receive queue is only four deep,
        hence the small gap between writes in _write().
        """
        steps = list(steps)
        if len(steps) > SEQ_STEPS_PER_MOTOR:
            raise MotionServiceError(
                f"Sequence of {len(steps)} steps exceeds the firmware's "
                f"{SEQ_STEPS_PER_MOTOR} per motor"
            )

        self._write(CMD_SEQ_CLEAR, [cfg.channel])
        for speed, ms, end_action in steps:
            hi, lo = _int16_bytes(-speed if cfg.invert else speed)
            self._write(
                CMD_SEQ_PUSH,
                [cfg.channel, hi, lo, hi, lo, (ms >> 8) & 0xFF, ms & 0xFF, end_action],
                retry=False,   # a retried push could double a step
            )
        self._write(CMD_SEQ_START, [cfg.channel, 1])

    def _set_speed(self, motor_name: str, counts: int) -> None:
        cfg = self.motors[motor_name]
        if cfg.invert:
            counts = -counts
        hi, lo = _int16_bytes(counts)
        self._write(CMD_SET, [cfg.channel, hi, lo])
        self._at_neutral[motor_name] = counts == 0

    def _hard_stop_all(self) -> None:
        """Coast every motor with one write; also clears running sequences."""
        if self._bus is None:
            return
        self._write(CMD_STOP_ALL)
        for name in self._at_neutral:
            self._at_neutral[name] = True

    def _arm(self, enabled: bool) -> None:
        self._require_init()
        self._arm_locked(enabled)

    def _arm_locked(self, enabled: bool) -> None:
        self._write(CMD_ENABLE, [1 if enabled else 0])
        self._armed = bool(enabled)

    def _set_pwm_frequency_locked(self, hz: int) -> None:
        self._write(CMD_SET_FREQ, [(hz >> 8) & 0xFF, hz & 0xFF])

    def _motor(self, motor_name: str) -> MotorConfig:
        cfg = self.motors.get(motor_name)
        if cfg is None:
            raise ValueError(f"Unknown motor '{motor_name}'")
        return cfg

    def _require_init(self) -> None:
        if not self._initialized:
            raise MotionServiceError("MotionService.initialize() must be called first")

    def _sleep_with_cancel(self, duration_s: float) -> None:
        end = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < end:
            if self._cancel_motion.is_set() or self._stop_worker.is_set():
                return
            time.sleep(0.005)

    # ---------------------------------------------------------------------
    # I2C transport
    # ---------------------------------------------------------------------
    def _write(self, opcode: int, payload: Iterable[int] = (), *, retry: bool = True) -> None:
        """
        Send one command to the coprocessor.

        A zero-argument command goes out as a bare byte, which the firmware
        queues as a one-byte command — only bytes >= 0x30 are read selectors.
        """
        data = [int(b) & 0xFF for b in payload]
        attempts = 1 + (self.i2c_retries if retry else 0)
        last_error: Optional[Exception] = None

        with self._bus_lock:
            if self._bus is None:
                raise MotionServiceError("I2C bus is not open")

            for _ in range(max(1, attempts)):
                try:
                    if data:
                        self._bus.write_i2c_block_data(self.address, opcode, data)
                    else:
                        self._bus.write_byte(self.address, opcode)
                    # The ATtiny executes commands from loop(), not the ISR,
                    # and its receive queue is four deep. A short gap keeps a
                    # burst (clear/push/push/start) from overrunning it.
                    if self.command_gap_s > 0:
                        time.sleep(self.command_gap_s)
                    return
                except OSError as exc:
                    last_error = exc
                    self._bus_errors += 1
                    time.sleep(0.002)

        raise MotionServiceError(
            f"I2C write 0x{opcode:02X} to 0x{self.address:02X} failed: {last_error}"
        )

    def _read_register(self, selector: int, length: int = 8) -> Optional[bytes]:
        """
        Select a register, then read it.

        Deliberately two transactions with a STOP between them: the firmware
        latches the selector in its receive ISR, so giving it a completed write
        first removes any dependency on how the slave handles a repeated START.
        """
        with self._bus_lock:
            if self._bus is None:
                return None
            try:
                self._bus.write_byte(self.address, selector)
                if self.command_gap_s > 0:
                    time.sleep(self.command_gap_s)
                message = i2c_msg.read(self.address, length)
                self._bus.i2c_rdwr(message)
                return bytes(list(message))
            except OSError:
                self._bus_errors += 1
                return None

    def _read_status(self, force: bool = False) -> Optional[dict]:
        """Read and decode REG_STATUS, caching the result for status()."""
        if self._bus is None:
            return None

        raw = self._read_register(REG_STATUS)
        if raw is None or len(raw) < 8:
            return None if force else self._telemetry

        flags = raw[0]
        telemetry = {
            "output_enabled": bool(flags & 0x01),
            "standby":        bool(flags & 0x02),
            "pi_present":     bool(flags & 0x04),
            "pca_ok":         bool(flags & 0x08),
            "any_moving":     bool(flags & 0x10),
            "sequencer_mask": raw[1],
            "firmware":       f"{raw[2] >> 4}.{raw[2] & 0x0F}",
            "i2c_errors":     raw[3],
            "temperature_c":  round(_int16_be(raw, 4) / 16.0, 2),
            "uptime_s":       (raw[6] << 8) | raw[7],
        }

        self._armed = telemetry["output_enabled"]
        self._telemetry = telemetry
        self._telemetry_at = time.monotonic()
        return telemetry

    def _close_bus(self) -> None:
        with self._bus_lock:
            if self._bus is not None:
                try:
                    self._bus.close()
                except Exception:
                    pass
                self._bus = None

    # ---------------------------------------------------------------------
    # Pure helpers
    # ---------------------------------------------------------------------
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


def _percent_to_counts(percent: float) -> int:
    """Percent 0-100 at the API boundary -> the firmware's 0-255 counts."""
    magnitude = max(0.0, min(100.0, float(percent)))
    return int(round(magnitude * SPEED_COUNTS_MAX / 100.0))


def _int16_bytes(value: int) -> Tuple[int, int]:
    """Clamp to the firmware's range and split into big-endian two's complement."""
    value = max(-SPEED_COUNTS_MAX, min(SPEED_COUNTS_MAX, int(value)))
    raw = value & 0xFFFF
    return (raw >> 8) & 0xFF, raw & 0xFF


def _int16_be(raw: bytes, offset: int) -> int:
    value = (raw[offset] << 8) | raw[offset + 1]
    return value - 0x10000 if value & 0x8000 else value


def _to_ms(seconds: float) -> int:
    """Seconds -> the sequencer's uint16 millisecond field."""
    return max(0, min(STEP_MS_MAX, int(round(max(0.0, seconds) * 1000.0))))
