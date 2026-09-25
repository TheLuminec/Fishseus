# Access Point Module

Thin local Wi-Fi access point service for Fishseus using `hostapd` + `dnsmasq`.
Hosts the isolated 2.4 GHz WPA2 **Fishseus** network that the Nerf turret
controller and ESP32-CAM join directly. Ported from the turret repo's standalone
`access_point.py` with the same network design, now run by the orchestrator.

**Responsibilities:** bring the AP up / down on a dedicated interface, serve DHCP
(with reserved MAC → IP leases), and report daemon state and connected clients.

**Non-responsibilities:** no internet sharing, by design — no NAT, no IP
forwarding, and DHCP advertises neither a gateway nor DNS. Clients reach this
host at `address` (or via mDNS / Avahi, e.g. `turret.local`) and nothing else.
No turret or camera logic.

## Configuration (`AccessPointConfig`, `"access_point"` in `fish_config.json`)

| Field              | Default                          | Purpose                                                   |
| ------------------ | -------------------------------- | --------------------------------------------------------- |
| `enabled`          | `false`                          | Orchestrator key: start the service at all.               |
| `ssid`             | `"Fishseus"`                     | Network name.                                             |
| `passphrase`       | `""`                             | Leave empty; read from `config/secrets.json`. Must match the ESP32 firmware. |
| `interface`        | `""`                             | Wireless interface (e.g. `wlan1` for the AC1300). Empty = auto if only one. |
| `country_code`     | `"US"`                           | Regulatory domain.                                        |
| `channel`          | `6`                              | 2.4 GHz channel (1/6/11 don't overlap).                   |
| `address`/`prefix` | `192.168.50.1` / `24`            | This host's address on the island.                        |
| `dhcp_start`/`dhcp_end` | `.50` – `.150`              | DHCP pool.                                                |
| `dhcp_lease`       | `"12h"`                          | Lease time.                                               |
| `reserved`         | turret `.51`, camera `.52`       | MAC → IP static leases.                                   |
| `stop_on_shutdown` | `false`                          | Keep the AP up when Fishseus exits (the turret needs it). |
| `use_sudo`         | `true`                           | Run privileged commands via `sudo -n` when not root.      |
| `run_dir`          | `<root>/tmp/access_point`        | Generated configs, pidfiles, hostapd log.                 |
| `leases_file`      | `/var/lib/misc/dnsmasq.leases`   | Read by `clients()` for IPs / hostnames.                  |

## Lifecycle API

- `initialize()` – validates config; adopts an AP that is already running (it
  outlives a fish restart), otherwise calls `start()`.
- `shutdown()` – stops the AP only if `stop_on_shutdown` is set.
- `reset()` – stops and restarts the AP, re-applying config.
- `status()` – `{enabled, service, running, daemons, ssid, interface, address, clients, last_error}`.

## Domain API

- `start()` / `stop()` – bring the AP up / down (idempotent). `start()` rolls
  back a partial bring-up and reports the hostapd log tail on failure.
- `is_running()`, `stations()` (associated MACs), `clients()` (MAC + lease IP / hostname).
- `dry_run()` – the hostapd / dnsmasq configs `start()` would write (passphrase masked).
- `list_wireless_interfaces()` – to pick `interface`.

## Assistant tools

`network_clients` – "who's on the Fishseus network?"

## Setup on the Pi

```bash
sudo apt install hostapd dnsmasq iw
iw dev                                  # find the AC1300 (onboard is usually wlan0)
cp config/secrets.example.json config/secrets.json   # set access_point.passphrase
```

Set `"access_point": {"enabled": true, "interface": "wlan1", ...}` in
`fish_config.json`.

Fishseus runs as a normal user, so give it passwordless sudo for exactly the
commands the service uses (`sudo visudo -f /etc/sudoers.d/fishseus-ap`, replace
`fish` with the user):

```
fish ALL=(root) NOPASSWD: /usr/sbin/hostapd, /usr/sbin/dnsmasq, /usr/sbin/ip, /usr/bin/ip, /usr/sbin/sysctl, /usr/bin/kill, /usr/bin/nmcli, /usr/bin/systemctl stop hostapd, /usr/bin/systemctl stop dnsmasq
```

Disable the distro units so they don't fight over the interface at boot:
`sudo systemctl disable --now hostapd dnsmasq`.

If you were running the turret repo's `access_point.py`, stop it first
(`sudo python access_point.py down`); both would otherwise manage the same
interface.

## Requirements

- Linux with `hostapd`, `dnsmasq`, `iw`, `ip`.
- A wireless adapter whose driver supports AP mode (`iw list` shows `* AP`). The
  RTL8812AU AC1300 needs an AP-capable driver such as the 8812au DKMS package.
