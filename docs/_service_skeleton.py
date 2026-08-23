"""
example_service.py

Thin local <purpose> service for Fishseus using <backend>.

Reference skeleton for the Fishseus service style — copy this into a new
`<name>/<name>_service.py`, rename the classes, and fill in the domain methods.
See docs/service_template.md for the full standard. This file is documentation:
it is not imported by the app.

Responsibilities:
- <the one or two things this service owns>

Non-responsibilities:
- No audio capture / motion / assistant / LLM logic (as applicable)

Example:
    svc = ExampleService()
    svc.initialize()
    svc.do_thing("hello")
    svc.shutdown()

Orchestrator usage:
    svc = ExampleService(ExampleConfig(**config.get("example", {})))
    svc.initialize()
    ...
    svc.shutdown()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from services import Service, ServiceConfig, ServiceError, ROOT_DIR


class ExampleServiceError(ServiceError):
    pass


@dataclass(frozen=True)
class ExampleConfig(ServiceConfig):
    module_name: str = "example"

    # Anchor paths to ROOT_DIR so cwd never matters.
    data_dir: Path = ROOT_DIR / "tmp" / "example"
    timeout_s: float = 30.0

    def validate(self) -> bool:
        if self.timeout_s <= 0:
            raise ExampleServiceError(f"timeout_s must be positive: {self.timeout_s}")
        return True


class ExampleService(Service):
    def __init__(self, config: ExampleConfig = ExampleConfig()) -> None:
        self.config = config
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.config.validate()  # raises ExampleServiceError on any problem
        if self._initialized:
            return
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        # ...acquire resources / start threads here...
        self._initialized = True

    def shutdown(self) -> None:
        # ...release resources / stop threads here; must be safe to call twice...
        self._initialized = False

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "service": "ok" if self._initialized else "uninitialized",
        }

    def reset(self) -> bool:
        self.shutdown()
        self.initialize()
        return True

    # ------------------------------------------------------------------
    # Domain API
    # ------------------------------------------------------------------
    def do_thing(self, text: str) -> str:
        if not self._initialized:
            raise ExampleServiceError("initialize() must be called first")
        return text.upper()
