"""
bluetooth_service.py

Thin local Bluetooth speaker service for Fishseus using BlueZ (`bluetoothctl`).

Responsibilities:
- Power the adapter on and (optionally) reconnect a default speaker at start-up
- Scan for, pair, trust, connect, disconnect, and forget audio devices
- Report which devices are known / connected

Non-responsibilities:
- No audio routing: PipeWire / PulseAudio (or bluez-alsa) owns the A2DP sink.
  Once a speaker is connected it becomes the default sink, so TTS (`aplay -D
  default`) and Spotify (raspotify) follow it automatically.
- No voice / assistant / LLM logic

Example:
    bt = BluetoothService(BluetoothConfig(default_speaker="JBL Flip 5"))
    bt.initialize()
    bt.pair("JBL")          # scan + pair + trust + connect
    bt.disconnect()
    bt.shutdown()

Orchestrator usage:
    bt = BluetoothService(BluetoothConfig(**config.get("bluetooth", {})))
    bt.initialize()          # powers on, reconnects default_speaker in background
    ...
    bt.shutdown()
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional

from services import Service, ServiceConfig, ServiceError


class BluetoothServiceError(ServiceError):
    pass


_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


def is_mac(value: str) -> bool:
    return bool(_MAC_RE.match(value.strip()))


def parse_devices(text: str) -> list[tuple[str, str]]:
    """(mac, name) pairs from `bluetoothctl devices` output."""
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) >= 2 and parts[0] == "Device" and is_mac(parts[1]):
            out.append((parts[1].upper(), parts[2] if len(parts) > 2 else parts[1]))
    return out


def parse_info(text: str) -> dict:
    """Key fields from `bluetoothctl info <mac>` / `bluetoothctl show` output."""
    info: dict = {}
    for line in text.splitlines():
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key in {"Name", "Alias", "Icon"}:
            info[key.lower()] = value
        elif key in {"Paired", "Trusted", "Connected", "Powered"}:
            info[key.lower()] = value.lower() == "yes"
    return info


@dataclass(frozen=True)
class BluetoothConfig(ServiceConfig):
    module_name: str = "bluetooth"

    # MAC or (partial) name of the speaker to reconnect at start-up / by default.
    default_speaker: str = ""
    auto_connect: bool = True
    scan_timeout_s: float = 10.0
    command_timeout_s: float = 20.0
    connect_retries: int = 2
    bluetoothctl_binary: str = "bluetoothctl"

    def validate(self) -> bool:
        if self.scan_timeout_s <= 0:
            raise BluetoothServiceError(f"scan_timeout_s must be positive: {self.scan_timeout_s}")
        if self.command_timeout_s <= 0:
            raise BluetoothServiceError(f"command_timeout_s must be positive: {self.command_timeout_s}")
        if self.connect_retries < 1:
            raise BluetoothServiceError(f"connect_retries must be >= 1: {self.connect_retries}")
        return True


class BluetoothService(Service):
    def __init__(self, config: BluetoothConfig = BluetoothConfig()) -> None:
        self.config = config
        self._initialized = False
        # bluetoothctl sessions interleave badly; run one command at a time.
        self._lock = threading.RLock()
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.config.validate()
        if self._initialized:
            return
        if platform.system() != "Linux":
            raise BluetoothServiceError(
                f"Bluetooth control needs Linux + BlueZ; this host is {platform.system()}")
        if not shutil.which(self.config.bluetoothctl_binary):
            raise BluetoothServiceError(
                f"{self.config.bluetoothctl_binary} not found (sudo apt install bluez)")

        self._ctl("power", "on")
        self._initialized = True

        if self.config.auto_connect and self.config.default_speaker:
            # A speaker that is off takes a while to time out — don't block start-up.
            threading.Thread(
                target=self._auto_connect, name="BluetoothAutoConnect", daemon=True
            ).start()

    def shutdown(self) -> None:
        # Leave the speaker connected: other audio (Spotify) may still be using it.
        self._initialized = False

    def status(self) -> dict:
        info: dict = {}
        connected: list[str] = []
        if self._initialized:
            try:
                info = parse_info(self._ctl("show", check=False))
                connected = [d["name"] for d in self.connected_devices()]
            except BluetoothServiceError as exc:
                self._last_error = str(exc)
        return {
            "enabled": self.enabled,
            "service": "ok" if self._initialized else "uninitialized",
            "powered": info.get("powered", False),
            "connected": connected,
            "default_speaker": self.config.default_speaker,
            "last_error": self._last_error,
        }

    def reset(self) -> bool:
        self.shutdown()
        self.initialize()
        return True

    # ------------------------------------------------------------------
    # Domain API
    # ------------------------------------------------------------------
    def list_devices(self) -> list[dict]:
        """Every device BlueZ knows about (paired or recently discovered)."""
        self._require_init()
        devices = []
        for mac, name in parse_devices(self._ctl("devices", check=False)):
            info = parse_info(self._ctl("info", mac, check=False))
            devices.append({
                "mac": mac,
                "name": info.get("alias") or info.get("name") or name,
                "icon": info.get("icon", ""),
                "paired": info.get("paired", False),
                "trusted": info.get("trusted", False),
                "connected": info.get("connected", False),
            })
        return devices

    def connected_devices(self) -> list[dict]:
        return [d for d in self.list_devices() if d["connected"]]

    def scan(self, timeout_s: Optional[float] = None) -> list[dict]:
        """Discover nearby devices for timeout_s seconds; returns list_devices()."""
        self._require_init()
        timeout = float(timeout_s or self.config.scan_timeout_s)
        with self._lock:
            self._ctl("--timeout", str(int(timeout)), "scan", "on",
                      check=False, timeout=timeout + 10)
        return self.list_devices()

    def pair(self, target: str) -> dict:
        """Find a device by MAC or name (scanning if needed), then pair + trust + connect."""
        self._require_init()
        with self._lock:
            device = self._find(target)
            if device is None:
                self.scan()
                device = self._find(target)
            if device is None:
                raise BluetoothServiceError(
                    f"no Bluetooth device matching '{target}' found — is it in pairing mode?")
            mac = device["mac"]
            if not device["paired"]:
                out = self._ctl("pair", mac, check=False)
                if "Pairing successful" not in out and "AlreadyExists" not in out:
                    raise BluetoothServiceError(
                        f"pairing with {device['name']} failed: {_last_line(out)}")
            if not device["trusted"]:
                self._ctl("trust", mac, check=False)  # lets it auto-reconnect later
            return self.connect(mac)

    def connect(self, target: str = "") -> dict:
        """Connect an already-paired device (default_speaker when target is empty)."""
        self._require_init()
        with self._lock:
            device = self._resolve_paired(target)
            if device["connected"]:
                return device
            out = ""
            for _ in range(self.config.connect_retries):
                out = self._ctl("connect", device["mac"], check=False)
                if "Connection successful" in out:
                    device["connected"] = True
                    self._last_error = None
                    return device
            self._last_error = f"connect to {device['name']} failed: {_last_line(out)}"
            raise BluetoothServiceError(
                f"could not connect to {device['name']} — is it switched on? ({_last_line(out)})")

    def disconnect(self, target: str = "") -> list[str]:
        """Disconnect one device (by MAC/name) or, with no target, every connected device."""
        self._require_init()
        with self._lock:
            if target:
                targets = [self._resolve_paired(target)]
            else:
                targets = self.connected_devices()
            for device in targets:
                self._ctl("disconnect", device["mac"], check=False)
            return [d["name"] for d in targets]

    def forget(self, target: str) -> str:
        """Unpair / remove a device from BlueZ."""
        self._require_init()
        with self._lock:
            device = self._find(target)
            if device is None:
                raise BluetoothServiceError(f"no known Bluetooth device matching '{target}'")
            self._ctl("remove", device["mac"], check=False)
            return device["name"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _auto_connect(self) -> None:
        try:
            device = self.connect(self.config.default_speaker)
            print(f"[bluetooth] Connected to {device['name']}", flush=True)
        except BluetoothServiceError as exc:
            print(f"[bluetooth] Auto-connect skipped: {exc}", flush=True)

    def _find(self, target: str) -> Optional[dict]:
        """Match a MAC exactly, else a case-insensitive name substring."""
        target = target.strip()
        devices = self.list_devices()
        if is_mac(target):
            return next((d for d in devices if d["mac"] == target.upper()), None)
        needle = target.lower()
        exact = [d for d in devices if d["name"].lower() == needle]
        if exact:
            return exact[0]
        partial = [d for d in devices if needle in d["name"].lower()]
        # Prefer paired devices when a partial name is ambiguous.
        partial.sort(key=lambda d: not d["paired"])
        return partial[0] if partial else None

    def _resolve_paired(self, target: str) -> dict:
        target = (target or self.config.default_speaker).strip()
        if target:
            device = self._find(target)
            if device is None or not device["paired"]:
                raise BluetoothServiceError(
                    f"'{target}' is not paired — pair it first")
            return device
        paired = [d for d in self.list_devices() if d["paired"]]
        if len(paired) == 1:
            return paired[0]
        if not paired:
            raise BluetoothServiceError("no paired Bluetooth speakers")
        names = ", ".join(d["name"] for d in paired)
        raise BluetoothServiceError(f"several paired devices ({names}) — say which one")

    def _ctl(self, *args: str, check: bool = True, timeout: Optional[float] = None) -> str:
        cmd = [self.config.bluetoothctl_binary, *args]
        with self._lock:
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=timeout or self.config.command_timeout_s,
                )
            except FileNotFoundError as exc:
                raise BluetoothServiceError(f"{cmd[0]} not found (sudo apt install bluez)") from exc
            except subprocess.TimeoutExpired as exc:
                raise BluetoothServiceError(f"`{' '.join(cmd)}` timed out") from exc
        if check and proc.returncode != 0:
            raise BluetoothServiceError(
                f"`{' '.join(cmd)}` failed ({proc.returncode}): {_last_line(proc.stderr or proc.stdout)}")
        return proc.stdout

    def _require_init(self) -> None:
        if not self._initialized:
            raise BluetoothServiceError("initialize() must be called first")


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else "no output"
