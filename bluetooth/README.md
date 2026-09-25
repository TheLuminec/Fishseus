# Bluetooth Module

Thin local Bluetooth speaker service for Fishseus using BlueZ's `bluetoothctl`.
Powers the adapter on, reconnects a default speaker at start-up, and lets the
fish pair / connect / disconnect speakers by voice.

**Responsibilities:** adapter power, scan, pair + trust + connect, disconnect,
forget, and reporting known / connected devices.

**Non-responsibilities:** no audio routing. PipeWire (Raspberry Pi OS Bookworm's
default) makes a connected A2DP speaker the default sink, so anything playing to
`default` — Piper TTS and raspotify — follows it automatically. No assistant or
LLM logic.

## Configuration (`BluetoothConfig`, `"bluetooth"` in `fish_config.json`)

| Field                 | Default          | Purpose                                                       |
| --------------------- | ---------------- | ------------------------------------------------------------- |
| `enabled`             | `false`          | Orchestrator key: start the service at all.                   |
| `default_speaker`     | `""`             | MAC or (partial) name used when no speaker is named.          |
| `auto_connect`        | `true`           | Reconnect `default_speaker` at start-up (background thread).  |
| `scan_timeout_s`      | `10.0`           | How long `scan()` / `pair()` discover for.                    |
| `command_timeout_s`   | `20.0`           | Timeout for each `bluetoothctl` call.                         |
| `connect_retries`     | `2`              | Connect attempts before giving up.                            |
| `bluetoothctl_binary` | `"bluetoothctl"` | Path to `bluetoothctl`.                                       |

## Lifecycle API

- `initialize()` – validates config, checks for Linux + `bluetoothctl`, powers the
  adapter on, and starts the default-speaker reconnect in the background.
- `shutdown()` – marks the service stopped; the speaker is left connected so
  Spotify keeps playing.
- `reset()` – `shutdown()` then `initialize()`.
- `status()` – `{enabled, service, powered, connected, default_speaker, last_error}`.

## Domain API

- `list_devices()` – every device BlueZ knows: `{mac, name, icon, paired, trusted, connected}`.
- `connected_devices()` – the connected subset.
- `scan(timeout_s=None)` – discover nearby devices, then `list_devices()`.
- `pair(target)` – find by MAC or name (scanning if needed), pair, trust, connect.
- `connect(target="")` – connect a paired device; empty = `default_speaker`, or
  the only paired device.
- `disconnect(target="")` – one device, or everything connected.
- `forget(target)` – remove the pairing.

Names match case-insensitively, exact first, then substring ("jbl" finds
"JBL Flip 5").

## Assistant tools

`connect_speaker(name)`, `pair_speaker(name)`, `disconnect_speaker(name)`.

## Usage

```python
bt = BluetoothService(BluetoothConfig(default_speaker="JBL Flip 5"))
bt.initialize()
bt.pair("JBL")            # speaker must be in pairing mode
bt.disconnect()
bt.connect()              # default_speaker
bt.shutdown()
```

## Requirements

- Linux with BlueZ (`sudo apt install bluez`) and a Bluetooth adapter.
- A2DP audio: PipeWire with `libspa-0.2-bluetooth` (default on Bookworm desktop
  images; on Lite: `sudo apt install pipewire wireplumber libspa-0.2-bluetooth`).
- For TTS to follow the speaker, set `tts.audio_device` to `"default"` (the
  PipeWire sink) instead of a fixed `plughw:` card.
- The Fishseus user must be in the `bluetooth` group (`sudo usermod -aG bluetooth $USER`).
