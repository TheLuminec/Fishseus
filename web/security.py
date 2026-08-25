#!/usr/bin/env python3
"""
Access control and hardening for the Fishseus web control panel.

The control panel drives physical hardware (motors), can execute registered
tools, and exposes the assistant's memory.  When the panel is published to the
internet through a Cloudflare Tunnel it MUST NOT be reachable without
authentication.  This module implements that gate.

Deployment model (see AskUserQuestion decisions):
  * The Flask app binds to loopback (127.0.0.1) only — nothing is exposed
    directly; `cloudflared` reaches it locally and Cloudflare terminates TLS.
  * Cloudflare Access (Zero Trust) authenticates users at the edge and injects
    a signed JWT (`Cf-Access-Jwt-Assertion`).  We verify that JWT here so the
    app is safe even if the tunnel is misconfigured or someone reaches
    cloudflared directly.

Because cloudflared connects to the app over loopback, tunnel traffic and
genuinely-local traffic both appear to originate from 127.0.0.1 — the source IP
cannot tell them apart.  What *can*: cloudflared injects `Cf-Ray` /
`Cf-Connecting-Ip` headers that a direct loopback client cannot forge against a
127.0.0.1-bound socket.  So the rule is:

  * Request carries Cloudflare headers  -> REQUIRE a valid Access JWT (fail-closed).
  * Request is direct loopback          -> allowed, unless a local token is set.

Settings are read from environment variables first (preferred for secrets,
since config/fish_config.json is committed to git), then the `web` config
section as a non-secret fallback.

    FISHSEUS_AUTH_ENABLED          "1"/"0"       (default: enabled)
    FISHSEUS_CF_ACCESS_TEAM_DOMAIN yourteam.cloudflareaccess.com
    FISHSEUS_CF_ACCESS_AUD         <Application Audience (AUD) tag>
    FISHSEUS_LOCAL_TOKEN           <shared secret for direct/local API calls>
    FISHSEUS_BIND_HOST             127.0.0.1
"""

from __future__ import annotations

import hmac
import os
import threading
from typing import Optional

# PyJWT (+ cryptography) is only needed when Cloudflare Access verification is
# actually used.  Import lazily/gracefully so the server still runs for local
# development without the dependency; if Access verification is required but the
# library is missing we fail CLOSED (deny) rather than open.
try:
    import jwt  # type: ignore[import-untyped]
    from jwt import PyJWKClient  # type: ignore[import-untyped]

    _JWT_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    jwt = None  # type: ignore[assignment]
    PyJWKClient = None  # type: ignore[assignment]
    _JWT_AVAILABLE = False


# ---------------------------------------------------------------------
# Settings resolution (env var takes precedence over config)
# ---------------------------------------------------------------------

def _env(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _bool_env(name: str) -> Optional[bool]:
    raw = _env(name)
    if raw is None:
        return None
    return raw.lower() in ("1", "true", "yes", "on")


def resolve_settings(web_conf: dict) -> dict:
    """Merge env vars (preferred) with the `web` config section."""
    web_conf = web_conf or {}

    auth_enabled = _bool_env("FISHSEUS_AUTH_ENABLED")
    if auth_enabled is None:
        auth_enabled = bool(web_conf.get("auth_enabled", True))

    return {
        "auth_enabled": auth_enabled,
        "bind_host": _env("FISHSEUS_BIND_HOST")
        or str(web_conf.get("bind_host", "127.0.0.1")),
        "team_domain": (
            _env("FISHSEUS_CF_ACCESS_TEAM_DOMAIN")
            or str(web_conf.get("cf_access_team_domain", "") or "")
        ).replace("https://", "").replace("http://", "").strip("/ "),
        "aud": _env("FISHSEUS_CF_ACCESS_AUD")
        or str(web_conf.get("cf_access_aud", "") or ""),
        "local_token": _env("FISHSEUS_LOCAL_TOKEN")
        or str(web_conf.get("local_bypass_token", "") or ""),
    }


# ---------------------------------------------------------------------
# Cloudflare Access JWT verification
# ---------------------------------------------------------------------

# One PyJWKClient per team domain; it fetches and caches the signing keys.
_jwks_clients: dict[str, "PyJWKClient"] = {}
_jwks_lock = threading.Lock()


def _get_jwks_client(team_domain: str) -> "PyJWKClient":
    certs_url = f"https://{team_domain}/cdn-cgi/access/certs"
    with _jwks_lock:
        client = _jwks_clients.get(team_domain)
        if client is None:
            # lifespan: refresh the key set periodically so rotations are picked up.
            client = PyJWKClient(certs_url, cache_keys=True, lifespan=3600)
            _jwks_clients[team_domain] = client
        return client


def _extract_access_token(request) -> str:
    """The Access JWT is in a header, or the CF_Authorization cookie."""
    token = request.headers.get("Cf-Access-Jwt-Assertion", "")
    if not token:
        token = request.cookies.get("CF_Authorization", "")
    return token


def verify_access_jwt(request, team_domain: str, aud: str) -> tuple[bool, str]:
    """Return (ok, reason).  ok is True only for a fully-valid Access JWT."""
    if not _JWT_AVAILABLE:
        # Required for verification but unavailable -> fail closed.
        return False, "PyJWT/cryptography not installed on server"

    token = _extract_access_token(request)
    if not token:
        return False, "missing Cloudflare Access token"

    try:
        signing_key = _get_jwks_client(team_domain).get_signing_key_from_jwt(token)
        jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=aud,
            issuer=f"https://{team_domain}",
        )
        return True, "ok"
    except Exception as exc:  # invalid signature, aud, issuer, expiry, etc.
        return False, f"invalid Access token: {exc}"


# ---------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------

def _came_through_cloudflare(request) -> bool:
    return bool(
        request.headers.get("Cf-Ray")
        or request.headers.get("Cf-Connecting-Ip")
    )


def authorize(request, web_conf: dict) -> tuple[bool, str, int]:
    """
    Decide whether a request may proceed.

    Returns (allowed, message, status_code).  message/status are only
    meaningful when allowed is False.
    """
    s = resolve_settings(web_conf)

    # Explicit opt-out for trusted LAN / development.
    if not s["auth_enabled"]:
        return True, "", 200

    # Shared local token — usable from anywhere it's presented correctly.
    token = s["local_token"]
    supplied = request.headers.get("X-Fishseus-Token", "")
    if token and hmac.compare_digest(supplied, token):
        return True, "", 200

    access_configured = bool(s["team_domain"]) and bool(s["aud"])

    if _came_through_cloudflare(request):
        # Anything arriving via the tunnel must be authenticated by Access.
        if not access_configured:
            return (
                False,
                "Server rejects tunnel traffic: Cloudflare Access is not "
                "configured (set FISHSEUS_CF_ACCESS_TEAM_DOMAIN and "
                "FISHSEUS_CF_ACCESS_AUD).",
                503,
            )
        ok, reason = verify_access_jwt(request, s["team_domain"], s["aud"])
        if ok:
            return True, "", 200
        return False, f"Forbidden: {reason}", 403

    # Direct loopback (local) access.
    if token:
        # A local token is configured but was not supplied/correct.
        return False, "Forbidden: missing or invalid local token", 403
    return True, "", 200


# ---------------------------------------------------------------------
# CSRF: same-origin check for state-changing requests
# ---------------------------------------------------------------------

_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def csrf_ok(request) -> bool:
    """
    Reject cross-site state-changing requests.

    Once a Cloudflare Access session cookie exists in the browser, a malicious
    third-party page could otherwise trigger authenticated POSTs (driving
    motors, running tools) that ride that cookie.  The panel's own fetch() calls
    are same-origin and send a matching Origin header; a cross-site request
    carries a foreign Origin.  Requests with no Origin (curl, native clients)
    are left to the auth gate.
    """
    if request.method not in _MUTATING_METHODS:
        return True
    origin = request.headers.get("Origin")
    if not origin:
        return True
    from urllib.parse import urlparse

    return urlparse(origin).netloc == request.host


# ---------------------------------------------------------------------
# Security response headers
# ---------------------------------------------------------------------

# All JS and CSS are inline in fishseus_ui.html, so 'unsafe-inline' is required.
# Google Fonts are pulled in via @import in styles.css.
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "object-src 'none'"
)

_SECURITY_HEADERS = {
    "Content-Security-Policy": _CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


def apply_security_headers(response):
    for key, val in _SECURITY_HEADERS.items():
        response.headers.setdefault(key, val)
    return response


# ---------------------------------------------------------------------
# Startup diagnostics
# ---------------------------------------------------------------------

def startup_report(web_conf: dict) -> list[str]:
    """Human-readable lines describing the effective security posture."""
    s = resolve_settings(web_conf)
    lines: list[str] = []
    if not s["auth_enabled"]:
        lines.append(
            "[web] *** AUTH DISABLED *** - every endpoint is open. Do NOT expose "
            "this to the internet. Set FISHSEUS_AUTH_ENABLED=1 before publishing."
        )
        return lines

    if s["team_domain"] and s["aud"]:
        if not _JWT_AVAILABLE:
            lines.append(
                "[web] Cloudflare Access is configured but PyJWT/cryptography is "
                "NOT installed - tunnel traffic will be DENIED. "
                "Run: pip install 'PyJWT[crypto]'"
            )
        else:
            lines.append(
                f"[web] Cloudflare Access enforced (team={s['team_domain']}, "
                "aud set). Tunnel traffic requires a valid Access JWT."
            )
    else:
        lines.append(
            "[web] Cloudflare Access NOT configured - tunnel traffic will be "
            "REJECTED (fail-closed). Direct loopback access "
            + ("requires the local token." if s["local_token"] else "is allowed.")
        )
    if s["local_token"]:
        lines.append("[web] Local token gate is active for direct/loopback calls.")
    return lines
