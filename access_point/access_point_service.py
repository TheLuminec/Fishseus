"""
access_point_service.py

Thin local Wi-Fi access point service for Fishseus using hostapd + dnsmasq.

Hosts the isolated 2.4 GHz WPA2 "Fishseus" network that the Nerf turret
controller and ESP32-CAM join directly. Ported from the turret repo's
standalone `access_point.py`; the network design is unchanged.

Responsibilities:
- Bring up / tear down the AP on a dedicated wireless interface (hostapd)
- Serve DHCP on it, including reserved MAC -> IP leases (dnsmasq)
- Report daemon state and associated stations

Non-responsibilities:
- No internet sharing, by design: no NAT, no IP forwarding, and DHCP advertises
  neither a gateway nor a DNS server — a true island. Clients reach this host
  at `address` (or via mDNS / Avahi) and nothing else.
- No turret / camera logic

Privileges: hostapd, dnsmasq, and `ip` need root while Fishseus runs as a normal
user. When not root, privileged commands are run through `sudo -n`, so the Pi
needs a NOPASSWD sudoers rule for them (see access_point/README.md). Without it
initialize() fails fast instead of prompting.

Example:
    ap = AccessPointService(AccessPointConfig(interface="wlan1"))
    ap.initialize()          # starts the AP (or adopts one already running)
    print(ap.stations())
    ap.stop()

Orchestrator usage:
    ap = AccessPointService(AccessPointConfig(**config.get("access_point", {})))
    ap.initialize()
    ...
    ap.shutdown()            # leaves the AP up unless stop_on_shutdown=True
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from services import ROOT_DIR, Service, ServiceConfig, ServiceError, load_secrets


class AccessPointServiceError(ServiceError):
    pass


DAEMONS = ("hostapd", "dnsmasq")
REQUIRED_TOOLS = ("hostapd", "dnsmasq", "ip", "iw")


@dataclass(frozen=True)
class AccessPointConfig(ServiceConfig):
    module_name: str = "access_point"

    ssid: str = "Fishseus"
    # Empty = read config/secrets.json ("access_point" section). Must match the
    # Fishseus entry in the ESP32 firmware's NETWORKS[] list.
    passphrase: str = ""
    interface: str = ""               # empty = auto-detect (errors if ambiguous)
    country_code: str = "US"          # regulatory domain; gates 2.4 GHz channels
    channel: int = 6                  # 1/6/11 are the non-overlapping ones
    address: str = "192.168.50.1"     # this host's IP = the AP "gateway"
    prefix: int = 24
    dhcp_start: str = "192.168.50.50"
    dhcp_end: str = "192.168.50.150"
    dhcp_lease: str = "12h"
    # MAC -> IP static leases (e.g. the turret and camera).
    reserved: dict[str, str] = field(default_factory=dict)

    # The ESP32s depend on this network, so by default it outlives a fish restart.
    stop_on_shutdown: bool = False
    use_sudo: bool = True
    run_dir: Path = ROOT_DIR / "tmp" / "access_point"
    # dnsmasq's default lease database on Debian / Raspberry Pi OS.
    leases_file: Path = Path("/var/lib/misc/dnsmasq.leases")

    def validate(self) -> bool:
        if not 1 <= len(self.ssid) <= 32:
            raise AccessPointServiceError("ssid must be 1-32 characters")
        if self.passphrase and not 8 <= len(self.passphrase) <= 63:
            raise AccessPointServiceError("passphrase must be 8-63 characters (WPA2)")
        if not 1 <= self.channel <= 14:
            raise AccessPointServiceError("channel must be a 2.4 GHz channel (1-14)")
        if not 8 <= self.prefix <= 30:
            raise AccessPointServiceError(f"prefix must be 8-30: {self.prefix}")
        for label, ip in (("address", self.address),
                          ("dhcp_start", self.dhcp_start), ("dhcp_end", self.dhcp_end)):
            if not is_ipv4(ip):
                raise AccessPointServiceError(f"{label} is not a valid IPv4 address: {ip!r}")
        for mac, ip in self.reserved.items():
            if not is_mac(mac):
                raise AccessPointServiceError(f"reserved: not a valid MAC address: {mac!r}")
            if not is_ipv4(ip):
                raise AccessPointServiceError(f"reserved: not a valid IPv4 address: {ip!r}")
        return True


# ----------------------------------------------------------------------
# Pure helpers (no I/O; unit-testable anywhere)
# ----------------------------------------------------------------------

def is_ipv4(s: str) -> bool:
    parts = str(s).split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def is_mac(s: str) -> bool:
    parts = str(s).split(":")
    return len(parts) == 6 and all(
        len(p) == 2 and all(c in "0123456789abcdefABCDEF" for c in p) for p in parts)


def netmask(prefix: int) -> str:
    bits = (0xFFFFFFFF << (32 - prefix)) & 0xFFFFFFFF
    return ".".join(str((bits >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def render_hostapd(cfg: AccessPointConfig, iface: str, passphrase: str) -> str:
    """hostapd.conf for a 2.4 GHz (hw_mode=g) WPA2-PSK AP."""
    lines = [
        "# Generated by access_point_service.py - edit fish_config.json instead.",
        f"interface={iface}",
        "driver=nl80211",
        f"ssid={cfg.ssid}",
        f"country_code={cfg.country_code}",
        "ieee80211d=1",
        "hw_mode=g",              # 2.4 GHz band (ESP32 is 2.4 GHz only)
        f"channel={cfg.channel}",
        "ieee80211n=1",
        "wmm_enabled=1",
        "auth_algs=1",
        "macaddr_acl=0",
        "ignore_broadcast_ssid=0",
        "wpa=2",
        "wpa_key_mgmt=WPA-PSK",
        "rsn_pairwise=CCMP",
        f"wpa_passphrase={passphrase}",
    ]
    return "\n".join(lines) + "\n"


def render_dnsmasq(cfg: AccessPointConfig, iface: str) -> str:
    """dnsmasq.conf: DHCP only, no DNS, no gateway advertised."""
    lines = [
        "# Generated by access_point_service.py - edit fish_config.json instead.",
        f"interface={iface}",
        "bind-interfaces",
        "except-interface=lo",
        "port=0",                # disable dnsmasq's DNS server (DHCP only)
        f"dhcp-range={cfg.dhcp_start},{cfg.dhcp_end},{netmask(cfg.prefix)},{cfg.dhcp_lease}",
        "dhcp-option=3",         # router: empty -> advertise NO gateway
        "dhcp-option=6",         # dns: empty -> advertise NO resolver
        "dhcp-authoritative",
    ]
    for mac, ip in sorted(cfg.reserved.items()):
        lines.append(f"dhcp-host={mac.lower()},{ip}")
    return "\n".join(lines) + "\n"


def parse_station_dump(text: str) -> list[str]:
    """MAC addresses of associated clients from `iw dev <iface> station dump`."""
    return [line.split()[1] for line in text.splitlines()
            if line.strip().startswith("Station ") and len(line.split()) > 1]


def parse_ap_capable(iw_list_text: str) -> bool:
    """True if `iw list` advertises AP mode under supported interface modes."""
    in_modes = False
    for line in iw_list_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Supported interface modes"):
            in_modes = True
            continue
        if in_modes:
            if stripped.startswith("*"):
                if stripped[1:].strip() == "AP":
                    return True
            elif stripped:
                in_modes = False
    return False


def parse_leases(text: str) -> dict[str, dict]:
    """MAC -> {ip, hostname} from a dnsmasq leases file."""
    leases: dict[str, dict] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and is_mac(parts[1]):
            leases[parts[1].lower()] = {
                "ip": parts[2],
                "hostname": "" if parts[3] == "*" else parts[3],
            }
    return leases


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------

class AccessPointService(Service):
    def __init__(self, config: AccessPointConfig = AccessPointConfig()) -> None:
        self.config = config
        self._initialized = False
        self._iface: Optional[str] = None
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.config.validate()
        if self._initialized:
            return
        if self.is_running():
            # Already up (e.g. survived a fish restart) — adopt it.
            self._iface = self._state_iface()
        else:
            self.start()
        self._initialized = True

    def shutdown(self) -> None:
        if self.config.stop_on_shutdown and self._initialized:
            try:
                self.stop()
            except AccessPointServiceError as exc:
                print(f"[access_point] stop failed: {exc}", flush=True)
        self._initialized = False

    def status(self) -> dict:
        alive = {name: self._pid_alive(name) for name in DAEMONS}
        stations: list[str] = []
        iface = self._iface or self._state_iface()
        if iface and all(alive.values()):
            try:
                stations = self.stations()
            except AccessPointServiceError as exc:
                self._last_error = str(exc)
        return {
            "enabled": self.enabled,
            "service": "ok" if self._initialized else "uninitialized",
            "running": all(alive.values()),
            "daemons": alive,
            "ssid": self.config.ssid,
            "interface": iface,
            "address": self.config.address,
            "clients": len(stations),
            "last_error": self._last_error,
        }

    def reset(self) -> bool:
        # A reset should actually re-apply config, so restart the AP itself.
        self._initialized = False
        self.stop()
        self.initialize()
        return True

    # ------------------------------------------------------------------
    # Domain API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Bring the AP up on the configured interface. Idempotent."""
        if self.is_running():
            return
        iface = self._preflight()
        passphrase = self._passphrase()
        run_dir = self.config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "iface").write_text(iface)

        self._release_interface(iface)
        self._priv(["ip", "link", "set", iface, "down"])
        self._priv(["ip", "addr", "flush", "dev", iface])
        self._priv(["ip", "link", "set", iface, "up"])
        self._priv(["ip", "addr", "add", f"{self.config.address}/{self.config.prefix}",
                    "dev", iface])
        # Isolation: never forward between this island and the main network.
        self._priv(["sysctl", "-q", "-w", "net.ipv4.ip_forward=0"], check=False)

        hostapd_conf = run_dir / "hostapd.conf"
        # Create with 0600 before writing: the file holds the passphrase.
        hostapd_conf.touch(mode=0o600, exist_ok=True)
        os.chmod(hostapd_conf, 0o600)
        hostapd_conf.write_text(render_hostapd(self.config, iface, passphrase))
        dnsmasq_conf = run_dir / "dnsmasq.conf"
        dnsmasq_conf.write_text(render_dnsmasq(self.config, iface))

        try:
            # Both daemonise themselves and write a pidfile; a non-zero exit
            # means they failed to bind / parse config.
            self._priv(["hostapd", "-B",
                        "-P", str(self._pidfile("hostapd")),
                        "-f", str(run_dir / "hostapd.log"),
                        str(hostapd_conf)])
            self._priv(["dnsmasq",
                        f"--conf-file={dnsmasq_conf}",
                        f"--pid-file={self._pidfile('dnsmasq')}"])
        except AccessPointServiceError as exc:
            self._last_error = f"{exc}\n{self._log_tail('hostapd')}"
            self.stop()  # roll back a partial bring-up
            raise AccessPointServiceError(self._last_error) from exc

        time.sleep(1.0)  # hostapd can still die after daemonising (driver refusal)
        if not self.is_running():
            self._last_error = f"AP daemons exited after start:\n{self._log_tail('hostapd')}"
            self.stop()
            raise AccessPointServiceError(self._last_error)
        self._iface = iface
        self._last_error = None

    def stop(self) -> None:
        """Stop both daemons and hand the interface back to NetworkManager."""
        if platform.system() != "Linux":
            return
        for name in DAEMONS:
            pid = self._read_pid(name)
            if pid is not None and self._pid_alive(name):
                self._priv(["kill", "-TERM", str(pid)], check=False)
                for _ in range(30):
                    if not self._pid_alive(name):
                        break
                    time.sleep(0.1)
            # run_dir is ours, so we can unlink root-owned pidfiles in it.
            self._pidfile(name).unlink(missing_ok=True)
        iface = self._iface or self._state_iface() or self.config.interface
        if iface and Path(f"/sys/class/net/{iface}").exists():
            self._priv(["ip", "addr", "flush", "dev", iface], check=False)
            self._priv(["ip", "link", "set", iface, "down"], check=False)
            if shutil.which("nmcli"):
                self._priv(["nmcli", "device", "set", iface, "managed", "yes"], check=False)
        self._iface = None

    def is_running(self) -> bool:
        return all(self._pid_alive(name) for name in DAEMONS)

    def stations(self) -> list[str]:
        """MACs of clients currently associated with the AP."""
        iface = self._iface or self._state_iface()
        if not iface or not shutil.which("iw"):
            return []
        return parse_station_dump(_run(["iw", "dev", iface, "station", "dump"], check=False))

    def clients(self) -> list[dict]:
        """Associated clients enriched with their DHCP lease (ip / hostname)."""
        try:
            leases = parse_leases(self.config.leases_file.read_text())
        except OSError:
            leases = {}
        out = []
        for mac in self.stations():
            lease = leases.get(mac.lower(), {})
            out.append({"mac": mac, "ip": lease.get("ip", ""), "hostname": lease.get("hostname", "")})
        return out

    def dry_run(self) -> str:
        """The hostapd / dnsmasq configs start() would write (passphrase masked)."""
        try:
            iface = self._resolve_interface()
        except AccessPointServiceError:
            iface = "wlan0"
        return (f"# ----- hostapd.conf (iface {iface}) -----\n"
                f"{render_hostapd(self.config, iface, '********')}\n"
                f"# ----- dnsmasq.conf (iface {iface}) -----\n"
                f"{render_dnsmasq(self.config, iface)}")

    @staticmethod
    def list_wireless_interfaces() -> list[str]:
        ifaces: list[str] = []
        if shutil.which("iw"):
            for line in _run(["iw", "dev"], check=False).splitlines():
                line = line.strip()
                if line.startswith("Interface "):
                    ifaces.append(line.split(None, 1)[1])
        if not ifaces:
            base = Path("/sys/class/net")
            if base.is_dir():
                ifaces = [p.name for p in sorted(base.iterdir())
                          if (p / "wireless").exists() or (p / "phy80211").exists()]
        return ifaces

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _preflight(self) -> str:
        if platform.system() != "Linux":
            raise AccessPointServiceError(
                f"the AP runs on Linux (the Pi) only; this host is {platform.system()}")
        missing = [t for t in REQUIRED_TOOLS if not shutil.which(t)]
        if missing:
            raise AccessPointServiceError(
                f"missing required tool(s): {missing} (sudo apt install hostapd dnsmasq iw)")
        if self._need_sudo():
            probe = subprocess.run(["sudo", "-n", "true"], capture_output=True)
            if probe.returncode != 0:
                raise AccessPointServiceError(
                    "passwordless sudo is not configured for the AP commands — "
                    "see access_point/README.md")
        iface = self._resolve_interface()
        if not Path(f"/sys/class/net/{iface}").exists():
            raise AccessPointServiceError(f"interface {iface!r} not found")
        if parse_ap_capable(_run(["iw", "list"], check=False)) is False:
            raise AccessPointServiceError(
                f"the driver for {iface} does not advertise AP mode (`iw list`); "
                "an RTL8812AU AC1300 needs an AP-capable driver such as the 8812au DKMS package")
        return iface

    def _resolve_interface(self) -> str:
        if self.config.interface:
            return self.config.interface
        ifaces = self.list_wireless_interfaces()
        if not ifaces:
            raise AccessPointServiceError("no wireless interface found; set access_point.interface")
        if len(ifaces) > 1:
            raise AccessPointServiceError(
                f"multiple wireless interfaces {ifaces}; set access_point.interface "
                "(the onboard radio is usually wlan0)")
        return ifaces[0]

    def _passphrase(self) -> str:
        passphrase = self.config.passphrase or load_secrets("access_point").get("passphrase", "")
        if not 8 <= len(passphrase) <= 63:
            raise AccessPointServiceError(
                "AP passphrase missing or not 8-63 characters — set access_point.passphrase "
                "in config/secrets.json (see config/secrets.example.json)")
        return passphrase

    def _release_interface(self, iface: str) -> None:
        """Stop anything else that might hold the interface, best-effort."""
        if shutil.which("systemctl"):
            for svc in DAEMONS:
                self._priv(["systemctl", "stop", svc], check=False)  # distro copies
        if shutil.which("nmcli"):
            self._priv(["nmcli", "device", "set", iface, "managed", "no"], check=False)

    def _need_sudo(self) -> bool:
        return self.config.use_sudo and hasattr(os, "geteuid") and os.geteuid() != 0

    def _priv(self, cmd: list[str], check: bool = True) -> str:
        if self._need_sudo():
            cmd = ["sudo", "-n", *cmd]
        return _run(cmd, check=check)

    def _pidfile(self, name: str) -> Path:
        return self.config.run_dir / f"{name}.pid"

    def _read_pid(self, name: str) -> Optional[int]:
        try:
            return int(self._pidfile(name).read_text().strip())
        except (OSError, ValueError):
            return None

    def _pid_alive(self, name: str) -> bool:
        pid = self._read_pid(name)
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True   # exists, owned by root (started via sudo)
        except (ProcessLookupError, OSError):
            return False
        return True

    def _state_iface(self) -> Optional[str]:
        f = self.config.run_dir / "iface"
        return f.read_text().strip() if f.exists() else None

    def _log_tail(self, name: str, lines: int = 8) -> str:
        log = self.config.run_dir / f"{name}.log"
        try:
            return "\n".join(log.read_text(errors="ignore").strip().splitlines()[-lines:])
        except OSError:
            return ""


def _run(cmd: list[str], check: bool = True, timeout: int = 20) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise AccessPointServiceError(f"command not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AccessPointServiceError(f"`{' '.join(cmd)}` timed out") from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AccessPointServiceError(f"`{' '.join(cmd)}` failed ({result.returncode}): {detail}")
    return result.stdout
