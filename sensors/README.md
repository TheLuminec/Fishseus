# Sensors Module

Thin GPIO sensor-watcher service for Fishseus. Watches digital sensors (PIR,
door switches, buttons) on a background thread and fires a callback on each
trigger, with a per-sensor cooldown.

**Responsibilities:** watch pins, debounce with cooldowns, emit `SensorEvent`s on
the inactive→active edge.

**Non-responsibilities:** no decision-making about events (the
orchestrator/assistant decides), no motion/audio/LLM logic. Does **not** call
`GPIO.cleanup()` — motion/orchestrator owns GPIO teardown.

## Configuration (`SensorConfig`)

| Field             | Default | Purpose                                  |
| ----------------- | ------- | ---------------------------------------- |
| `module_name`     | `"sensors"` | Service key in the orchestrator config. |
| `poll_interval_s` | `0.05`  | How often pins are polled.               |
| `sensors`         | `[]`    | List of sensor entries (see below).      |

Each sensor entry: `{"name", "pin" (BCM), "active_high", "cooldown_s",
"description"}`.

`config.validate()` raises `SensorServiceError` on a non-positive
`poll_interval_s` or a sensor entry missing `pin`. It runs in `initialize()`.

## Lifecycle API

- `initialize()` – validate config, set up pins, start the watcher thread.
  Degrades gracefully (logs and stays disabled) if `RPi.GPIO` is unavailable or
  no sensors are configured.
- `shutdown()` – stop the watcher (leaves GPIO teardown to motion/orchestrator).
- `reset()` – `shutdown()` then `initialize()`.
- `status()` – `{enabled, service, watcher_alive, gpio_available, sensor_count}`.

## Sensor API

- `set_callback(fn)` – register the function fired (from the watcher thread) on
  each trigger, receiving a `SensorEvent` (`name`, `description`, `triggered_at`).
- `sensor_report()` – richer per-sensor state (`name`, `description`, `active`,
  `seconds_since_trigger`) for tools and the web UI. (Distinct from the summary
  `status()` dict.)

## How it works

- The watcher polls each pin every `poll_interval_s`, tracks last state, and fires
  the callback only on an inactive→active edge.
- Each sensor has its own `cooldown_s` so a busy hallway can't spam the fish.

## Usage

```python
def on_event(event):
    print(f"{event.name} triggered: {event.description}")

sensors = SensorService(SensorConfig(sensors=[
    {"name": "motion", "pin": 16, "active_high": True,
     "cooldown_s": 60, "description": "PIR near the fish"},
]))
sensors.set_callback(on_event)
sensors.initialize()
...
sensors.shutdown()
```

## Requirements

- A Raspberry Pi with `RPi.GPIO` and sensors wired to the configured BCM pins.
  Without GPIO the service simply stays disabled.
