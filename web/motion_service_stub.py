"""
motion_service_stub.py
======================

This module provides a minimal stand‑in for the motion service used
by the Fishseus project.  In the real system, the motion service
controls the GPIO pins on a Raspberry Pi to animate the mouth, tail
and body motors.  Here we simulate those functions so that the web
UI can trigger them without needing access to the hardware.

The stub maintains internal state (e.g. the last motor parameters) and
returns simple status strings when actions are performed.  You can
extend this stub to print to the console or log to a file for
debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class MotorSettings:
    in1: int
    in2: int
    en: int
    forward_speed: int
    reverse_speed: int
    neutral_return_time: float


@dataclass
class MotionConfig:
    pwm_frequency: int = 1000
    body_wiggle_time: float = 0.18
    tail_wiggle_time: float = 0.14
    mouth_open_time: float = 0.09
    mouth_close_time: float = 0.04
    envelope_window_s: float = 0.18
    motors: Dict[str, MotorSettings] = field(default_factory=lambda: {
        "mouth": MotorSettings(17, 27, 22, 82, 55, 0.04),
        "tail": MotorSettings(23, 24, 25, 72, 48, 0.03),
        "body": MotorSettings(5, 6, 12, 68, 45, 0.03),
    })


class MotionServiceStub:
    """A non‑blocking, no‑hardware motion controller stub.

    Methods on this class simulate the behaviour of the real motion
    service.  They return strings describing what would have been
    done.  All state is kept in memory; there is no persistence.
    """

    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()

    def update_config(self, data: Dict[str, Any]) -> None:
        """Update the internal configuration with new values.

        Only keys present in ``data`` will be updated.  Nested
        dictionaries (e.g. ``motors``) are updated by key.
        """
        for key, value in data.items():
            if hasattr(self.config, key):
                if key == "motors" and isinstance(value, dict):
                    # update individual motor settings
                    for m_name, m_vals in value.items():
                        if m_name in self.config.motors and isinstance(m_vals, dict):
                            motor = self.config.motors[m_name]
                            for attr, attr_val in m_vals.items():
                                if hasattr(motor, attr):
                                    setattr(motor, attr, attr_val)
                else:
                    setattr(self.config, key, value)

    def open_mouth(self, duration: float | None = None, speed: int | None = None) -> str:
        """Simulate opening the fish's mouth.

        Parameters
        ----------
        duration: float | None
            Optional override for how long the mouth should remain open (seconds).
            If not provided, the configured ``mouth_open_time`` will be used.
        speed: int | None
            Optional override for the forward PWM duty cycle to open the mouth.
            If not provided, the configured ``forward_speed`` for the mouth
            motor will be used.

        Returns
        -------
        str
            A message describing the simulated action.
        """
        # Use overrides when provided, otherwise fall back to config.
        motor = self.config.motors["mouth"]
        open_time = duration if duration is not None else self.config.mouth_open_time
        forward_speed = speed if speed is not None else motor.forward_speed
        return (
            f"Opening mouth (pins {motor.in1},{motor.in2},{motor.en}) at speed "
            f"{forward_speed} and holding for {open_time:.2f}s before closing."
        )

    def wiggle(self, cycles: int = 1) -> str:
        """Simulate a wiggle of the tail and body for a number of cycles."""
        body = self.config.motors["body"]
        tail = self.config.motors["tail"]
        return (
            f"Wiggling body (pins {body.in1},{body.in2},{body.en}) and tail (pins "
            f"{tail.in1},{tail.in2},{tail.en}) for {cycles} cycle(s)."
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the current motion configuration as a dict."""
        cfg = {
            "pwm_frequency": self.config.pwm_frequency,
            "body_wiggle_time": self.config.body_wiggle_time,
            "tail_wiggle_time": self.config.tail_wiggle_time,
            "mouth_open_time": self.config.mouth_open_time,
            "mouth_close_time": self.config.mouth_close_time,
            "envelope_window_s": self.config.envelope_window_s,
            "motors": {},
        }
        for name, motor in self.config.motors.items():
            cfg["motors"][name] = {
                "in1": motor.in1,
                "in2": motor.in2,
                "en": motor.en,
                "forward_speed": motor.forward_speed,
                "reverse_speed": motor.reverse_speed,
                "neutral_return_time": motor.neutral_return_time,
            }
        return cfg
