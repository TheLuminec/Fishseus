"""
spotify_auth.py — one-time Spotify authorisation for a headless Pi.

Spotify's OAuth needs a browser once. This prints the authorisation URL; open
it on any device, approve, then paste back the URL the browser was redirected
to (the page itself will fail to load — that's expected). The refresh token is
cached at SpotifyConfig.token_cache_path and SpotifyService renews it silently
from then on.

Usage (from the repo root):
    python -m spotify.spotify_auth
"""

from __future__ import annotations

import json
import sys

from services import ROOT_DIR
from spotify.spotify_service import SpotifyConfig, SpotifyServiceError, build_oauth


def main() -> int:
    config_file = ROOT_DIR / "config" / "fish_config.json"
    cfg: dict = {}
    if config_file.exists():
        cfg = dict(json.loads(config_file.read_text(encoding="utf-8")).get("spotify", {}))
    cfg.pop("enabled", None)

    try:
        oauth = build_oauth(SpotifyConfig(**cfg))
    except SpotifyServiceError as exc:
        print(f"spotify_auth: {exc}", file=sys.stderr)
        return 1

    print("1. Open this URL in a browser and approve access:\n")
    print(f"   {oauth.get_authorize_url()}\n")
    print("2. Paste the full URL you were redirected to (starts with "
          f"{oauth.redirect_uri}):")
    redirected = input("> ").strip()

    code = oauth.parse_response_code(redirected)
    if not code or code == redirected:
        print("spotify_auth: no ?code= found in that URL", file=sys.stderr)
        return 1
    try:
        oauth.get_access_token(code, as_dict=False, check_cache=False)
    except Exception as exc:
        print(f"spotify_auth: token exchange failed: {exc}", file=sys.stderr)
        return 1

    print(f"Authorised. Token cached at {SpotifyConfig(**cfg).token_cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
