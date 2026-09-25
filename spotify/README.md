# Spotify Module

Thin Spotify playback-control service for Fishseus using
[spotipy](https://spotipy.readthedocs.io/). Lets you ask the fish to play a song,
artist, album, or playlist, and to pause, skip, or change the volume. The music
is turned down automatically while the fish talks.

**Responsibilities:** catalogue search, starting playback, transport control,
volume, now-playing, and ducking around fish speech.

**Non-responsibilities:** no audio output of its own. The Spotify Web API only
*controls* a Spotify Connect device; on the Pi that device is
[raspotify](https://github.com/dtcooper/raspotify) (librespot), which plays to the
default sink — the Bluetooth speaker when one is connected. No interactive OAuth
at runtime (see [Authorisation](#authorisation)).

Requires **Spotify Premium**; the playback endpoints reject free accounts.

## Configuration (`SpotifyConfig`, `"spotify"` in `fish_config.json`)

| Field               | Default                             | Purpose                                                   |
| ------------------- | ----------------------------------- | --------------------------------------------------------- |
| `enabled`           | `false`                             | Orchestrator key: start the service at all.               |
| `client_id`         | `""`                                | Leave empty; read from `config/secrets.json`.             |
| `client_secret`     | `""`                                | Leave empty; read from `config/secrets.json`.             |
| `redirect_uri`      | `http://127.0.0.1:8888/callback`    | Must match the developer app's Redirect URI exactly.      |
| `token_cache_path`  | `<root>/data/spotify_token.json`    | Cached refresh token (gitignored).                        |
| `device_name`       | `"Fishseus"`                        | Connect device to play on (substring match). Empty = active device. |
| `market`            | `"from_token"`                      | Search market.                                            |
| `duck_percent`      | `30`                                | Music volume (% of current) while the fish speaks; `100` disables. |
| `request_timeout_s` | `10.0`                              | HTTP timeout per Web API call.                            |

Credentials are resolved in order: config field → `config/secrets.json`
(`"spotify": {"client_id", "client_secret"}`) → `SPOTIPY_CLIENT_ID` /
`SPOTIPY_CLIENT_SECRET` env vars. They are kept out of `fish_config.json`
because that file is committed and served by the web UI.

## Lifecycle API

- `initialize()` – validates config, loads the cached token (refreshing it if
  expired). Raises `SpotifyServiceError` if spotipy, credentials, or the token
  are missing; the orchestrator logs it and runs without Spotify.
- `shutdown()` – drops the client; music keeps playing.
- `reset()` – `shutdown()` then `initialize()`.
- `status()` – `{enabled, service, device_name, ducked, last_error}`.

## Domain API

- `play(query, kind="track")` – search and play the top match; `kind` is
  `track`, `album`, `artist`, or `playlist`. Returns e.g. `"Playing Bohemian Rhapsody by Queen"`.
- `pause()`, `resume()`, `next_track()`, `previous_track()`
- `set_volume(percent)` – 0–100.
- `now_playing()` – `"Playing X by Y"` / `"Paused on …"` / `"Nothing is playing"`.
- `is_playing()` – never raises.
- `duck()` / `unduck()` – lower / restore the volume; never raise. The
  orchestrator calls these around every spoken reply.

## Assistant tools

`play_music(query, kind)`, `pause_music`, `resume_music`, `skip_track`,
`previous_track`, `set_music_volume(percent)`, `now_playing`.

"Hey fish, play Bohemian Rhapsody by Queen" → `play_music("bohemian rhapsody queen")`.

## Authorisation

1. Create an app at <https://developer.spotify.com/dashboard>, add the Redirect
   URI `http://127.0.0.1:8888/callback`, and copy the client ID / secret into
   `config/secrets.json` (copy `config/secrets.example.json`).
2. On the Pi, once: `web/.venv/bin/python -m spotify.spotify_auth`. Open the
   printed URL on any device, approve, and paste back the URL the browser was
   redirected to (the page itself won't load — that's fine).
3. Set `"spotify": {"enabled": true}` and restart Fishseus.

## Requirements

- `spotipy` (in `web/requirements.txt`, the venv Fishseus runs in).
- Spotify Premium.
- A Connect device on the Pi: `curl -sL https://dtcooper.github.io/raspotify/install.sh | sh`,
  then set `LIBRESPOT_NAME="Fishseus"` in `/etc/raspotify/conf` so it matches
  `device_name`. Play something to it once from the Spotify app so it registers
  with your account.
