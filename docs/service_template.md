# Fishseus service template

Every service in Fishseus follows the same shape so the orchestrator
(`fishseus.py`) and the web UI can treat them uniformly. This document is the
standard; [`_service_skeleton.py`](_service_skeleton.py) is a copy-paste
starting point.

## The three base types (from `services.py`)

- **`ServiceError`** — base for all service exceptions. Each service defines its
  own `XServiceError(ServiceError)`.
- **`ServiceConfig`** — frozen dataclass base. Subclasses add `module_name` and a
  `validate()` that **raises** on any problem.
- **`Service`** — ABC with four lifecycle methods every service implements:
  `initialize() -> None`, `shutdown() -> None`, `status() -> dict`,
  `reset() -> bool`.

`services.py` also exports **`ROOT_DIR`** (the repo root); anchor all default
paths to it so a service works regardless of the current working directory.

## Checklist for a conforming service

1. **Module docstring** in the house format:
   - One-line summary (`Thin local X service for Fishseus using Y.`)
   - `Responsibilities:` bullet list
   - `Non-responsibilities:` bullet list (what it deliberately does *not* do)
   - `Example:` a minimal runnable snippet
   - `Orchestrator usage:` how `fishseus.py` drives it
2. **Imports** are package-style: `from services import Service, ServiceConfig,
   ServiceError, ROOT_DIR` and `from other.other_service import …`. No
   `sys.path` manipulation, no bare `from x_service import …`.
3. **Error class**: `class XServiceError(ServiceError): pass`.
4. **Config**: `@dataclass(frozen=True) class XConfig(ServiceConfig)` with
   `module_name: str = "x"` as the first field, all fields defaulted, paths
   anchored to `ROOT_DIR`, and a `validate(self) -> bool` that raises
   `XServiceError` on any problem and returns `True` otherwise.
5. **Service**: `class XService(Service)` with
   `__init__(self, config: XConfig = XConfig())` and all four lifecycle methods:
   - `initialize()` calls `self.config.validate()`, sets up resources, is
     idempotent (safe to call twice).
   - `shutdown()` tears everything down cleanly and never raises on a
     double-call.
   - `status()` returns a flat `dict` starting with `enabled` and `service`.
   - `reset()` is typically `shutdown()` then `initialize()`, returning `True`.
6. **No `if __name__ == "__main__":` demo runner** — keep modules import-only.
7. **A `README.md`** in the service directory (see structure below).

## Notes & conventions

- **Frozen inheritance:** because `ServiceConfig` is `frozen=True`, every
  subclass must also be `@dataclass(frozen=True)`, or class creation raises
  `TypeError`. Don't mutate config after construction.
- **Field order:** `module_name` (inherited default) comes first in the generated
  `__init__`, so prefer keyword construction (`XConfig(field=…)`) — positional
  args would fill `module_name` first.
- **`status()` returns a dict.** If a service also needs to expose richer data
  (e.g. a per-item report), give that its own method (`sensor_report()`), not
  `status()`.
- **Background threads / subprocesses:** guard start/stop with a lock, make the
  worker survive individual job failures (log and continue, never let an
  exception kill the thread), and surface a dead child's exit code/stderr in the
  raised error.
- **Optional dependencies** (GPIO, cameras): degrade gracefully — a service that
  can't start should log and leave itself disabled rather than crash the
  orchestrator.

## README structure

Each service `README.md` mirrors `tts/README.md` and `stt/README.md`:

1. Title + one-paragraph summary, with Responsibilities / Non-responsibilities.
2. **Configuration** table: field, default, purpose.
3. **Lifecycle API**: the four `Service` methods.
4. **Domain API**: the service-specific methods.
5. **How it works**: brief mechanism.
6. **Usage**: short runnable examples.
7. **Requirements**: binaries/hardware/models it needs.
