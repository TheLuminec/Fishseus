# Motion Module

Controls fish motors (mouth/tail/body) via GPIO and supports preset animation actions.

## Main API (`motion_service.py`)
- `MotionService.initialize()` – configures GPIO + PWM resources.
- `MotionService.wiggle(cycles=1)` – queues tail/body wiggle pattern.
- `MotionService.open_mouth()` – mouth pulse action.
- `MotionService.speak_placeholder(duration_s)` – synthetic speaking animation.
- `MotionService.speak_audio(wav_path)` – mouth motion driven by WAV amplitude envelope.
- `MotionService.shutdown()` – stops worker thread and resets GPIO.

## How it works
- Uses per-motor pin/speed config (`MotorConfig`).
- Runs motion queue on background worker for non-blocking actions.
- Applies short forward/reverse pulses and neutral return timing.
- For `speak_audio`, computes per-window audio levels and maps them to mouth movement.

## Typical usage
```python
motion.initialize()
motion.wiggle(cycles=2)
motion.speak_audio("tmp/tts/line.wav")
```
