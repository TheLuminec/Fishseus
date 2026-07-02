"""
sensor_service.py

GPIO sensor watcher for the Fishseus assistant.

Watches digital sensors (PIR motion detectors, door switches, buttons, ...)
on a background thread and fires a callback when one triggers.  Each sensor
has its own cooldown so a busy hallway doesn't spam the fish.

The orchestrator decides what to DO with events — this service only detects
and reports them.

Config shape (config/fish_config.json "sensors" section):
    {
      "enabled": true,
      "poll_interval_s": 0.05,
      "sensors": [
        {
          "name": "motion",
          "pin": 16,
          "active_high": true,
          "cooldown_s": 60,
          "description": "PIR motion detector near the fish"
        }
      ]
    }

Example:
    def on_event(event):
        print(f"{event.name} triggered!")

    sensors = SensorService(SensorConfig(sensors=[...]))
    sensors.initialize()
    sensors.set_callback(on_event)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except Exception:
    GPIO = None  # type: ignore[assignment]
    _GPIO_AVAILABLE = False


@dataclass(frozen=True)
class SensorEvent:
    name: str                # sensor name from config
    description: str         # human-readable description for the LLM
    triggered_at: float      # time.time() of the trigger


@dataclass(frozen=True)
class SensorConfig:
    poll_interval_s: float = 0.05
    # Each entry: name, pin (BCM), active_high, cooldown_s, description
    sensors: list[dict[str, Any]] = field(default_factory=list)


class SensorService:
    def __init__(self, config: SensorConfig = SensorConfig()) -> None:
        self.config = config
        self._callback: Optional[Callable[[SensorEvent], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._initialized = False
        # name -> last trigger time.monotonic(); used for cooldowns
        self._last_trigger: dict[str, float] = {}
        # name -> last observed pin state; trigger on inactive->active edge
        self._last_state: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        if self._initialized:
            return
        if not _GPIO_AVAILABLE:
            print("[SensorService] RPi.GPIO unavailable — sensors disabled")
            return
        if not self.config.sensors:
            print("[SensorService] No sensors configured")
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for sensor in self.config.sensors:
            pin = int(sensor.get("pin", -1))
            if pin < 0:
                continue
            GPIO.setup(pin, GPIO.IN)
            name = str(sensor.get("name", f"pin{pin}"))
            self._last_state[name] = False

        self._stop.clear()
        self._thread = threading.Thread(target=self._watch_loop, name="SensorWatcher", daemon=True)
        self._thread.start()
        self._initialized = True
        names = [s.get("name") for s in self.config.sensors]
        print(f"[SensorService] Watching sensors: {names}")

    def shutdown(self) -> None:
        if not self._initialized:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._initialized = False
        # GPIO.cleanup() is owned by motion_service / orchestrator shutdown —
        # do not call it here or we'd tear down the motor pins too.

    def set_callback(self, callback: Callable[[SensorEvent], None]) -> None:
        """Register the function fired (from the watcher thread) on each trigger."""
        self._callback = callback

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> list[dict[str, Any]]:
        """Current state of every sensor — for tools and the web UI."""
        now = time.monotonic()
        report = []
        for sensor in self.config.sensors:
            name = str(sensor.get("name", "?"))
            last = self._last_trigger.get(name)
            report.append({
                "name": name,
                "description": sensor.get("description", ""),
                "active": self._last_state.get(name, False),
                "seconds_since_trigger": round(now - last, 1) if last else None,
            })
        return report

    # ------------------------------------------------------------------
    # Watcher
    # ------------------------------------------------------------------

    def _watch_loop(self) -> None:
        while not self._stop.is_set():
            for sensor in self.config.sensors:
                try:
                    self._check_sensor(sensor)
                except Exception as exc:
                    print(f"[SensorService] sensor check failed: {exc}")
            time.sleep(self.config.poll_interval_s)

    def _check_sensor(self, sensor: dict[str, Any]) -> None:
        pin = int(sensor.get("pin", -1))
        if pin < 0:
            return

        name = str(sensor.get("name", f"pin{pin}"))
        active_high = bool(sensor.get("active_high", True))
        cooldown_s = float(sensor.get("cooldown_s", 60.0))

        raw = GPIO.input(pin)
        active = bool(raw) if active_high else not bool(raw)

        was_active = self._last_state.get(name, False)
        self._last_state[name] = active

        # Rising edge only.
        if not active or was_active:
            return

        now = time.monotonic()
        last = self._last_trigger.get(name, 0.0)
        if now - last < cooldown_s:
            return
        self._last_trigger[name] = now

        event = SensorEvent(
            name=name,
            description=str(sensor.get("description", name)),
            triggered_at=time.time(),
        )
        print(f"[SensorService] Trigger: {name}")
        if self._callback is not None:
            try:
                self._callback(event)
            except Exception as exc:
                print(f"[SensorService] callback failed: {exc}")


# ----------------------------------------------------------------------
# Manual test runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    service = SensorService(
        SensorConfig(
            sensors=[
                {"name": "motion", "pin": 16, "active_high": True,
                 "cooldown_s": 5, "description": "PIR motion detector"},
            ],
        )
    )
    service.set_callback(lambda e: print(f"EVENT: {e}"))
    service.initialize()
    try:
        print("Watching for 60s… wave at the sensor.")
        time.sleep(60)
    finally:
        service.shutdown()
