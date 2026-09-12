from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()

class ServiceError(RuntimeError):
    """Base class for service-related errors."""
    pass


@dataclass(frozen=True)
class ServiceConfig(ABC):
    """Abstract base class for service configuration."""
    module_name: str = ""

    def __post_init__(self) -> None:
        # Path-typed fields often arrive as strings — e.g. a JSON config splatted
        # in as SomeConfig(**cfg). Dataclasses don't coerce to the annotated type,
        # so normalise any Path field here (frozen -> object.__setattr__).
        for f in fields(self):
            if f.type is Path or f.type == "Path":
                value = getattr(self, f.name)
                if value is not None and not isinstance(value, Path):
                    object.__setattr__(self, f.name, Path(value))

    @abstractmethod
    def validate(self) -> bool:
        """Validate the configuration.  Return True if valid, False otherwise."""
        pass

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        d = dict(self.__dict__)
        d.pop("module_name", None)
        return {self.module_name: d}


class Service(ABC):
    """Abstract base class for services."""
    config: ServiceConfig
    enabled: bool = True

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the service."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the service."""
        pass

    @abstractmethod
    def status(self) -> dict:
        """Get the status of the service and service modules. \n
        Return a dict of status information.\n
        e.g. {"enabled": True, "service": "ok", "module1": "ok", "module2": "error"}"""
        pass

    @abstractmethod
    def reset(self) -> bool:
        """Reset the service.  Return True if successful, False otherwise."""
        pass

