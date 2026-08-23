# Motion Module

Thin motion-control service for a Billy Bass style fish (**Pi-only** — drives
GPIO via `RPi.GPIO`). Runs the mouth/tail/body H-bridge motors on a background
worker so orchestrator calls return immediately.

**Responsibilities:** PWM motor control, high-level motion helpers, mouth
animation from a WAV amplitude envelope, safe shutdown / emergency stop.

**Non-responsibilities:** no audio capture, STT, TTS, assistant, or LLM logic.

> This module imports `RPi.GPIO` at load time, so it only imports on the Pi.

## Configuration (`MotionConfig` / `MotorConfig`)

`MotionConfig` holds the service settings; `motors` maps a name to a `MotorConfig`
(H-bridge `in1`/`in2` + `en` PWM pins, plus speeds). Defaults to a built-in
3-motor pinout (`mouth`/`tail`/`body`).

| `MotionConfig` field | Default            | Purpose                                  |
| -------------------- | ------------------ | ---------------------------------------- |
| `module_name`        | `"motion"`         | Service key in the orchestrator config.  |
| `motors`             | built-in 3 motors  | `{name: MotorConfig}`.                   |
| `pwm_frequency`      | `1000`             | PWM frequency (Hz).                      |
| `body_wiggle_time`   | `0.18`             | Body pulse duration.                     |
| `tail_wiggle_time`   | `0.14`             | Tail pulse duration.                     |
| `mouth_open_time`    | `0.09`             | Mouth-open pulse duration.               |
| `mouth_close_time`   | `0.04`             | Mouth-close (reverse) pulse duration.    |
| `envelope_window_s`  | `0.18`             | Audio window for mouth animation.        |

| `MotorConfig` field   | Purpose                                          |
| --------------------- | ------------------------------------------------ |
| `in1`, `in2`, `en`    | H-bridge direction pins + PWM enable pin (BCM).  |
| `forward_speed`       | Forward PWM duty (0–100).                        |
| `reverse_speed`       | Reverse PWM duty (0–100).                        |
| `neutral_return_time` | Reverse pulse toward neutral after a move.       |

`config.validate()` raises `MotionServiceError` if no motors are configured. It
runs in `initialize()`.

## Lifecycle API

- `initialize()` – set up GPIO/PWM and start the motion worker (idempotent).
- `shutdown()` – stop the worker, hard-stop motors, `GPIO.cleanup()`.
- `reset()` – `shutdown()` then `initialize()`.
- `status()` – `{enabled, service, worker_alive, motors, queue_size}`.

## Motion API

Queued (non-blocking): `open_mouth()`, `wiggle(cycles, tail, body, speed_scale)`,
`speak_audio(wav_path)`, `speak_text_placeholder(duration_s)`, `stop_all()`.

Direct (for web-UI tuning, bypasses the queue): `direct_drive(motor, direction,
speed)`, `direct_stop(motor=None)`.

## How it works

- Orchestrator-facing methods enqueue commands; a background worker owns motor
  timing and sequencing, so calls never block.
- Moves are short forward pulses followed by a soft reverse toward neutral (the
  mechanism springs back on its own).
- `speak_audio` reads the WAV in `envelope_window_s` windows, estimates each
  window's level, and maps it to graded mouth movement.

## Usage

```python
motion = MotionService(MotionConfig())   # built-in default pinout
motion.initialize()
motion.wiggle(cycles=2)
motion.speak_audio("tmp/tts/line.wav")
motion.shutdown()
```

## Requirements

- A Raspberry Pi with `RPi.GPIO`, and H-bridge-driven motors wired to the
  configured BCM pins.
