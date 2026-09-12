# Motion Module

Thin motion-control service for a Billy Bass style fish (**Pi-only** — talks to
the Fishseus RPi HAT over I2C). Runs the mouth/tail/body motors on a background
worker so orchestrator calls return immediately.

**Responsibilities:** arm/disarm the HAT, drive motors over I2C, high-level
motion helpers, mouth animation from a WAV amplitude envelope, coprocessor
telemetry, safe shutdown / emergency stop.

**Non-responsibilities:** no audio capture, STT, TTS, assistant, or LLM logic.

> This module imports `smbus2` at load time and opens `/dev/i2c-N` in
> `initialize()`, so it is only useful on the Pi.

## The hardware it talks to

The Pi no longer drives motor pins. The HAT carries an ATtiny1614 motor
coprocessor that owns a PCA9685 and two TB6612FNG H-bridges:

```
Pi  --I2C(master)-->  ATtiny1614  --I2C-->  PCA9685 --> 2x TB6612 --> M1..M4
   GPIO2/GPIO3, addr 0x28
```

So motors are addressed by **channel** (`0..3` = M1..M4 on the silkscreen,
connectors U13/U14/U15/U16) rather than by IN1/IN2/EN GPIO numbers, and the
ATtiny's non-blocking step sequencer executes the actual pulse timing — a pulse
is millisecond-accurate even while Python is busy.

The protocol lives in the firmware's `pi_link.h`; the opcode constants at the
top of `motion_service.py` mirror it.

### Arming

Nothing moves until the board is armed. `/OE` has a pull-up, so at reset or
brown-out every PCA9685 output is forced low — and since channel 15 is the
shared TB6612 `STBY`, both bridges sit high-Z. `initialize()` arms the board
(`auto_enable`), `shutdown()` disarms it, and `emergency_stop()` drops the line
with a single write that does not depend on the motion queue being healthy.

## Configuration (`MotionConfig` / `MotorConfig`)

`MotionConfig` holds the service settings; `motors` maps a name to a
`MotorConfig`. Defaults to a built-in 3-motor map (`mouth`=M1, `tail`=M2,
`body`=M3).

| `MotionConfig` field   | Default    | Purpose                                        |
| ---------------------- | ---------- | ---------------------------------------------- |
| `module_name`          | `"motion"` | Service key in the orchestrator config.        |
| `motors`               | 3 motors   | `{name: MotorConfig}`.                         |
| `i2c_bus`              | `1`        | `/dev/i2c-1` = GPIO2/GPIO3.                    |
| `i2c_address`          | `0x28`     | The ATtiny's slave address.                    |
| `i2c_retries`          | `1`        | Retries per failed write.                      |
| `command_gap_s`        | `0.003`    | Gap between writes; the ATtiny queue is 4 deep.|
| `auto_enable`          | `True`     | Arm the kill line during `initialize()`.       |
| `heartbeat_interval_s` | `1.0`      | Idle status polls; `0` disables.               |
| `pwm_frequency`        | `1500`     | PCA9685 carrier, 24–1526 Hz.                   |
| `body_wiggle_time`     | `0.18`     | Body pulse duration.                           |
| `tail_wiggle_time`     | `0.14`     | Tail pulse duration.                           |
| `mouth_open_time`      | `0.09`     | Mouth-open pulse duration.                     |
| `mouth_close_time`     | `0.04`     | Mouth-close (reverse) pulse duration.          |
| `envelope_window_s`    | `0.18`     | Audio window for mouth animation.              |

| `MotorConfig` field   | Purpose                                              |
| --------------------- | ---------------------------------------------------- |
| `channel`             | `0..3` = M1..M4.                                     |
| `forward_speed`       | Forward speed, percent 0–100.                        |
| `reverse_speed`       | Reverse speed, percent 0–100.                        |
| `neutral_return_time` | Reverse pulse toward neutral after a move.           |
| `invert`              | Flip this motor's direction in software.             |

Speeds stay percent at this API's boundary (matching the web UI's sliders) and
are scaled to the firmware's `-255..255` counts on the wire.

`config.validate()` raises `MotionServiceError` for an empty motor map, a
channel outside `0..3`, two motors claiming the same channel, or a PWM
frequency the PCA9685 cannot produce. It runs in `initialize()`.

## Lifecycle API

- `initialize()` – open the bus, verify the bridge answers, set the carrier,
  arm the board, start the motion worker (idempotent).
- `shutdown()` – stop the worker, coast the motors, disarm, close the bus.
- `reset()` – `shutdown()` then `initialize()`.
- `status()` – service state plus a `bridge` dict of live telemetry:
  `output_enabled`, `standby`, `pi_present`, `pca_ok`, `any_moving`,
  `sequencer_mask`, `firmware`, `i2c_errors`, `temperature_c`, `uptime_s`.

## Motion API

Queued (non-blocking): `open_mouth()`, `wiggle(cycles, tail, body,
speed_scale)`, `speak_audio(wav_path)`, `speak_text_placeholder(duration_s)`.

Immediate: `stop_all()`, `emergency_stop()`, `arm(enabled)`,
`set_pwm_frequency(hz)`, `motor_speeds()`.

Direct (for web-UI tuning, bypasses the queue): `direct_drive(motor, direction,
speed)`, `direct_stop(motor=None)`.

## How it works

- Orchestrator-facing methods enqueue commands; a background worker owns
  sequencing, so calls never block. The HAT owns pulse timing.
- A move is a short forward pulse followed by a soft reverse toward neutral
  (the mechanism springs back on its own). Both steps are pushed to the
  coprocessor's sequencer as one program — clear, push, push, start — and the
  worker sleeps for the same span so queued motions stay in order.
- `speak_audio` reads the WAV in `envelope_window_s` windows, estimates each
  window's level, and maps it to graded mouth movement. The pulse runs *inside*
  the window and only the remainder is slept, so the mouth tracks the audio
  instead of drifting behind it.
- While idle, the worker polls `0x30` every `heartbeat_interval_s`. That read
  also refreshes the firmware's "the Pi is alive" timestamp, which is what
  `LINK_FAILSAFE_MS` watches if you turn it on in the firmware.

## Usage

```python
motion = MotionService(MotionConfig())   # built-in default channel map
motion.initialize()
motion.wiggle(cycles=2)
motion.speak_audio("tmp/tts/line.wav")
motion.shutdown()
```

## Requirements

- A Raspberry Pi with I2C enabled (`raspi-config`) and `smbus2` installed.
- The Fishseus RPi HAT, with the ATtiny1614 flashed from `attiny1614_bridge`.
- `i2cdetect -y 1` should show `0x28`. If it does not, the coprocessor is not
  running — check power and reflash before debugging anything on this side.
