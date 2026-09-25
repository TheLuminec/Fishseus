"""
spotify_service.py

Thin Spotify playback-control service for Fishseus using spotipy (Web API).

Responsibilities:
- Search the Spotify catalogue and start a track / album / artist / playlist
- Transport control: pause, resume, skip, previous, volume, now-playing
- Get the music out of the way while the fish speaks (lower or pause it),
  then restore it

Non-responsibilities:
- No audio output of its own: Spotify's Web API only *controls* a Spotify
  Connect device. On the Pi that device is librespot (e.g. raspotify), which
  plays to the default sink — the Bluetooth speaker when one is connected.
- No OAuth browser flow at runtime: run `python -m spotify.spotify_auth` once to
  cache a refresh token; the service then refreshes it silently.
- No voice / assistant / LLM logic

Requires Spotify Premium (the playback endpoints reject free accounts).

Example:
    sp = SpotifyService()
    sp.initialize()
    sp.play("bohemian rhapsody")
    sp.pause()
    sp.shutdown()

Orchestrator usage:
    sp = SpotifyService(SpotifyConfig(**config.get("spotify", {})))
    sp.initialize()          # raises SpotifyServiceError if not yet authorised
    sp.duck() / sp.unduck()  # around fish speech (lower/restore or pause/resume)
    sp.shutdown()
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from services import ROOT_DIR, Service, ServiceConfig, ServiceError, load_secrets

try:
    import spotipy
    from spotipy.cache_handler import CacheFileHandler
    from spotipy.oauth2 import SpotifyOAuth
except ImportError:  # optional dependency — service reports it at initialize()
    spotipy = None


class SpotifyServiceError(ServiceError):
    pass


SCOPES = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
KINDS = ("track", "album", "artist", "playlist")
DUCK_MODES = ("volume", "pause", "off")


@dataclass(frozen=True)
class SpotifyConfig(ServiceConfig):
    module_name: str = "spotify"

    # Credentials: leave empty to read config/secrets.json ("spotify" section),
    # falling back to the SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET env vars.
    client_id: str = ""
    client_secret: str = ""
    # Must exactly match a Redirect URI registered on the Spotify developer app.
    # Spotify no longer accepts "localhost" — use the loopback IP.
    redirect_uri: str = "http://127.0.0.1:8888/callback"
    token_cache_path: Path = ROOT_DIR / "data" / "spotify_token.json"

    # Spotify Connect device to play on (case-insensitive substring of its name,
    # e.g. the raspotify DEVICE_NAME). Empty = whichever device is active.
    device_name: str = "Fishseus"
    # ISO country code (e.g. "US"). Empty = omit it, and Spotify uses the
    # account's own country. Avoid "from_token": it needs the
    # user-read-private scope, which we don't request.
    market: str = ""
    # What happens to the music while the fish talks:
    #   "volume" - drop to duck_percent of the current level. Needs an output
    #              both librespot and aplay can open at once (the shared dmix
    #              "fishout" device, see spotify/README.md), otherwise aplay
    #              fails with "Device or resource busy".
    #   "pause"  - pause, then resume. Works with any audio setup: librespot
    #              releases the ALSA device when paused, so aplay can open it.
    #   "off"    - leave the music alone.
    duck_mode: str = "volume"
    duck_percent: int = 30
    request_timeout_s: float = 10.0

    def validate(self) -> bool:
        if self.duck_mode not in DUCK_MODES:
            raise SpotifyServiceError(f"duck_mode must be one of {DUCK_MODES}: {self.duck_mode!r}")
        if not 0 <= self.duck_percent <= 100:
            raise SpotifyServiceError(f"duck_percent must be 0-100: {self.duck_percent}")
        if self.request_timeout_s <= 0:
            raise SpotifyServiceError(f"request_timeout_s must be positive: {self.request_timeout_s}")
        if not self.redirect_uri.startswith(("http://", "https://")):
            raise SpotifyServiceError(f"redirect_uri must be an http(s) URL: {self.redirect_uri}")
        return True


def build_oauth(config: SpotifyConfig, *, open_browser: bool = False) -> "SpotifyOAuth":
    """SpotifyOAuth manager shared by the service and the one-off auth script."""
    if spotipy is None:
        raise SpotifyServiceError("spotipy is not installed (pip install spotipy)")
    secrets = load_secrets("spotify")
    client_id = config.client_id or secrets.get("client_id") or os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = (config.client_secret or secrets.get("client_secret")
                     or os.environ.get("SPOTIPY_CLIENT_SECRET"))
    if not client_id or not client_secret:
        raise SpotifyServiceError(
            "Spotify client_id / client_secret missing — add a \"spotify\" section to "
            "config/secrets.json (see config/secrets.example.json)")
    config.token_cache_path.parent.mkdir(parents=True, exist_ok=True)
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=config.redirect_uri,
        scope=SCOPES,
        cache_handler=CacheFileHandler(cache_path=str(config.token_cache_path)),
        open_browser=open_browser,
    )


class SpotifyService(Service):
    def __init__(self, config: SpotifyConfig = SpotifyConfig()) -> None:
        self.config = config
        self._initialized = False
        self._sp: Optional["spotipy.Spotify"] = None
        # Web and main threads both drive playback; keep requests (and the
        # duck/unduck volume bookkeeping) ordered.
        self._lock = threading.RLock()
        self._ducked_from: Optional[int] = None   # "volume" mode: level to restore
        self._paused_for_speech = False           # "pause" mode: resume afterwards
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.config.validate()
        if self._initialized:
            return
        oauth = build_oauth(self.config)
        # validate_token() refreshes an expired token from the cache; None means
        # nobody has run the interactive auth yet.
        if oauth.validate_token(oauth.cache_handler.get_cached_token()) is None:
            raise SpotifyServiceError(
                "Spotify is not authorised yet — run `python -m spotify.spotify_auth` once")
        self._sp = spotipy.Spotify(
            auth_manager=oauth,
            requests_timeout=self.config.request_timeout_s,
            retries=1,
        )
        self._initialized = True

    def shutdown(self) -> None:
        # Leave the music playing: the fish restarting shouldn't stop the party.
        self._sp = None
        self._ducked_from = None
        self._paused_for_speech = False
        self._initialized = False

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "service": "ok" if self._initialized else "uninitialized",
            "device_name": self.config.device_name,
            "ducked": self._ducked_from is not None or self._paused_for_speech,
            "last_error": self._last_error,
        }

    def reset(self) -> bool:
        self.shutdown()
        self.initialize()
        return True

    # ------------------------------------------------------------------
    # Domain API
    # ------------------------------------------------------------------
    def play(self, query: str, kind: str = "track") -> str:
        """Search for `query` and start playing the best match. Returns a description."""
        kind = (kind or "track").strip().lower()
        if kind not in KINDS:
            raise SpotifyServiceError(f"kind must be one of {KINDS}: {kind!r}")
        query = (query or "").strip()
        if not query:
            raise SpotifyServiceError("nothing to search for")

        with self._lock:
            sp = self._client()
            results = self._call(sp.search, q=query, type=kind, limit=1, market=self.config.market or None)
            items = [i for i in (results.get(f"{kind}s", {}).get("items") or []) if i]
            if not items:
                return f"Couldn't find any {kind} matching '{query}'"
            item = items[0]
            device_id = self._device_id()
            if kind == "track":
                self._call(sp.start_playback, device_id=device_id, uris=[item["uri"]])
            else:
                self._call(sp.start_playback, device_id=device_id, context_uri=item["uri"])
            # A fresh play starts at the device's own volume, and must not be
            # "resumed" again by a pending unduck().
            self._ducked_from = None
            self._paused_for_speech = False
            return f"Playing {describe(item, kind)}"

    def pause(self) -> str:
        with self._lock:
            self._call(self._client().pause_playback, device_id=self._device_id())
            self._paused_for_speech = False  # an explicit pause should stick
        return "Music paused"

    def resume(self) -> str:
        with self._lock:
            self._call(self._client().start_playback, device_id=self._device_id())
        return "Music resumed"

    def next_track(self) -> str:
        with self._lock:
            self._call(self._client().next_track, device_id=self._device_id())
        return "Skipped to the next track"

    def previous_track(self) -> str:
        with self._lock:
            self._call(self._client().previous_track, device_id=self._device_id())
        return "Back to the previous track"

    def set_volume(self, percent: int) -> str:
        percent = max(0, min(int(percent), 100))
        with self._lock:
            self._call(self._client().volume, percent, device_id=self._device_id())
            self._ducked_from = None
        return f"Music volume set to {percent} percent"

    def now_playing(self) -> str:
        with self._lock:
            current = self._call(self._client().current_playback, market=self.config.market or None)
        if not current or not current.get("item"):
            return "Nothing is playing"
        state = "Playing" if current.get("is_playing") else "Paused on"
        return f"{state} {describe(current['item'], 'track')}"

    def playback_state(self) -> dict:
        """Structured now-playing info for the web UI ({} when nothing is loaded)."""
        with self._lock:
            current = self._call(self._client().current_playback, market=self.config.market or None)
        if not current:
            return {}
        item = current.get("item") or {}
        device = current.get("device") or {}
        images = (item.get("album") or {}).get("images") or []
        return {
            "is_playing": bool(current.get("is_playing")),
            "title": item.get("name", ""),
            "artists": ", ".join(a.get("name", "") for a in item.get("artists") or [] if a),
            "album": (item.get("album") or {}).get("name", ""),
            # Spotify lists 640 / 300 / 64 px; the middle one suits the page.
            "image": images[min(1, len(images) - 1)]["url"] if images else "",
            "progress_ms": current.get("progress_ms") or 0,
            "duration_ms": item.get("duration_ms") or 0,
            "device": device.get("name", ""),
            "volume": device.get("volume_percent"),
            "shuffle": bool(current.get("shuffle_state")),
            "ducked": self._ducked_from is not None or self._paused_for_speech,
        }

    def devices(self) -> list[dict]:
        """Spotify Connect devices visible to the account."""
        with self._lock:
            found = self._call(self._client().devices).get("devices") or []
        return [{
            "id": d.get("id"),
            "name": d.get("name", ""),
            "type": d.get("type", ""),
            "is_active": bool(d.get("is_active")),
            "volume": d.get("volume_percent"),
        } for d in found]

    def transfer(self, device_id: str, play: bool = True) -> str:
        """Move playback to another Connect device."""
        with self._lock:
            self._call(self._client().transfer_playback, device_id, force_play=play)
        return "Playback moved"

    def set_shuffle(self, state: bool) -> str:
        with self._lock:
            self._call(self._client().shuffle, bool(state), device_id=self._device_id())
        return f"Shuffle {'on' if state else 'off'}"

    def is_playing(self) -> bool:
        """Cheap-ish check used by the orchestrator; never raises."""
        if not self._initialized:
            return False
        try:
            with self._lock:
                current = self._call(self._client().current_playback)
            return bool(current and current.get("is_playing"))
        except SpotifyServiceError:
            return False

    def duck(self) -> None:
        """Get the music out of the way while the fish speaks (see duck_mode).
        Never raises; no-op when nothing is playing."""
        if not self._initialized or self.config.duck_mode == "off":
            return
        if self.config.duck_mode == "volume" and self.config.duck_percent >= 100:
            return
        try:
            with self._lock:
                if self._ducked_from is not None or self._paused_for_speech:
                    return
                current = self._call(self._client().current_playback)
                if not current or not current.get("is_playing"):
                    return
                device = current.get("device") or {}
                if self.config.duck_mode == "pause":
                    self._call(self._client().pause_playback, device_id=device.get("id"))
                    self._paused_for_speech = True
                    return
                volume = device.get("volume_percent")
                if volume is None or not device.get("supports_volume", True):
                    return
                self._call(self._client().volume,
                           int(volume * self.config.duck_percent / 100),
                           device_id=device.get("id"))
                self._ducked_from = int(volume)
        except SpotifyServiceError as exc:
            print(f"[spotify] duck failed: {exc}", flush=True)

    def unduck(self) -> None:
        """Undo duck(): resume the music or restore its volume. Never raises."""
        with self._lock:
            volume, self._ducked_from = self._ducked_from, None
            paused, self._paused_for_speech = self._paused_for_speech, False
            if not self._initialized:
                return
            try:
                if paused:
                    self._call(self._client().start_playback)
                elif volume is not None:
                    self._call(self._client().volume, volume)
            except SpotifyServiceError as exc:
                print(f"[spotify] unduck failed: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _client(self) -> "spotipy.Spotify":
        if not self._initialized or self._sp is None:
            raise SpotifyServiceError("initialize() must be called first")
        return self._sp

    def _device_id(self) -> Optional[str]:
        """The configured Connect device's id, or None to use the active device."""
        devices = self._call(self._client().devices).get("devices") or []
        wanted = self.config.device_name.strip().lower()
        if wanted:
            for device in devices:
                if wanted in (device.get("name") or "").lower():
                    return device.get("id")
        if any(d.get("is_active") for d in devices):
            return None
        if wanted:
            raise SpotifyServiceError(
                f"Spotify device '{self.config.device_name}' is offline — is raspotify running?")
        raise SpotifyServiceError("no Spotify device is available to play on")

    def _call(self, fn, *args: Any, **kwargs: Any) -> Any:
        try:
            result = fn(*args, **kwargs)
        except spotipy.SpotifyException as exc:
            self._last_error = _friendly(exc)
            raise SpotifyServiceError(self._last_error) from exc
        except Exception as exc:  # network errors, token refresh failures
            self._last_error = f"Spotify request failed: {exc}"
            raise SpotifyServiceError(self._last_error) from exc
        self._last_error = None
        return result


def describe(item: dict, kind: str) -> str:
    """Human-friendly name for a search / playback item."""
    name = item.get("name", "something")
    if kind in {"track", "album"}:
        artists = ", ".join(a.get("name", "") for a in item.get("artists") or [] if a)
        return f"{name} by {artists}" if artists else name
    if kind == "playlist":
        return f"the playlist {name}"
    return name  # artist


def _friendly(exc: "spotipy.SpotifyException") -> str:
    status = getattr(exc, "http_status", None)
    reason = getattr(exc, "reason", None) or getattr(exc, "msg", "") or str(exc)
    if status == 403:
        return f"Spotify refused the request (Premium required?): {reason}"
    if status == 404:
        return f"No active Spotify device: {reason}"
    return f"Spotify error {status}: {reason}"
