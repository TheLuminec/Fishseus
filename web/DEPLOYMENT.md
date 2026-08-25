# Hosting the Fishseus control panel over Cloudflare

The control panel drives physical hardware, can execute registered tools, and
exposes the assistant's memory. Treat it as a privileged admin surface. This
guide gets it online safely behind a Cloudflare Tunnel + Cloudflare Access.

## Security model

- The app binds to **loopback only** (`127.0.0.1`) by default. Nothing is
  exposed directly to the LAN or internet; `cloudflared` reaches it locally.
- **Cloudflare Access** (Zero Trust) authenticates users at the edge and injects
  a signed JWT. The app **verifies that JWT** (`web/security.py`), so it stays
  safe even if the tunnel is misconfigured or cloudflared is reached directly.
- Requests arriving through Cloudflare (identified by the `Cf-Ray` /
  `Cf-Connecting-Ip` headers cloudflared injects) **must** carry a valid Access
  JWT — otherwise they are rejected **fail-closed**.
- Direct loopback requests (a shell on the Pi) stay allowed for local admin,
  unless you set a local token.
- State-changing requests are same-origin checked (CSRF), request bodies are
  capped at 1 MiB, and standard security headers + a CSP are sent on every
  response.

## 1. Install dependencies (on the Pi)

The Pi's Python is externally managed (PEP 668), so use a virtualenv for the web
dependencies. **Create it with `--system-site-packages`** — the web server
imports the hardware services (`RPi.GPIO`, `gpiozero`, `requests`), which are
installed as system packages; without that flag those services would import as
"unavailable".

```bash
python -m venv web/.venv --system-site-packages
web/.venv/bin/pip install -r web/requirements.txt
```

`PyJWT[crypto]` is required for Access verification. If it is missing while
Access is enabled, tunnel traffic is denied.

**Run everything with this interpreter.** Because `fishseus.py` imports the web
server in-process, the *orchestrator* also needs Flask — so launch it (or the
standalone panel) with the venv's Python:

```bash
web/.venv/bin/python fishseus.py --port 8000     # full assistant + web UI
web/.venv/bin/python web/web_server.py 8000      # standalone config panel
```

With `--system-site-packages`, hardware libs resolve from the system while
Flask / PyJWT / waitress resolve from the venv.

## 2. Create the Cloudflare Tunnel

```bash
cloudflared tunnel login
cloudflared tunnel create fishseus
# Route a hostname to the local app:
cloudflared tunnel route dns fishseus fish.example.com
```

Point the tunnel's ingress at the loopback app (config.yml):

```yaml
ingress:
  - hostname: fish.example.com
    service: http://127.0.0.1:8000
  - service: http_status:404
```

## 3. Protect it with Cloudflare Access

In the Cloudflare Zero Trust dashboard:

1. **Access → Applications → Add a self-hosted application.**
2. Application domain: `fish.example.com`.
3. Add a policy that allows only **your** email(s).
4. Open the application's settings and copy its **Application Audience (AUD)
   tag**. Your team domain is `https://<your-team>.cloudflareaccess.com`.

## 4. Configure the app

Set these as environment variables (preferred — `config/fish_config.json` is
committed to git, so keep secrets out of it):

```bash
export FISHSEUS_AUTH_ENABLED=1
export FISHSEUS_CF_ACCESS_TEAM_DOMAIN=your-team.cloudflareaccess.com
export FISHSEUS_CF_ACCESS_AUD=<the AUD tag from step 3>
# Optional: also require a token for direct loopback admin calls
# export FISHSEUS_LOCAL_TOKEN=$(openssl rand -hex 32)
```

The same knobs exist (non-secret) in `config/fish_config.json` under `"web"`:
`bind_host`, `auth_enabled`, `cf_access_team_domain`, `cf_access_aud`,
`devices_cache_ttl_s`. Environment variables win over the file.

At startup the server prints its effective posture, e.g.
`Cloudflare Access enforced (team=…, aud set)`.

## 5. (Recommended) Run under a real WSGI server

Flask's built-in server is fine for a single user behind the tunnel, but for a
production posture use waitress:

```bash
# from the repo root, so `web.web_server` is importable:
web/.venv/bin/waitress-serve --host 127.0.0.1 --port 8000 web.web_server:app
```

(The orchestrator `fishseus.py` still launches the Flask server in a background
thread; switch it to waitress if you want the whole stack production-grade.)

## Verifying

```bash
# Health check (unauthenticated, safe):
curl -s http://127.0.0.1:8000/healthz

# From the public hostname, without an Access session -> Cloudflare blocks you.
# Simulated tunnel request without a JWT is rejected by the app:
curl -s -H 'Cf-Ray: test' http://127.0.0.1:8000/api/status   # -> 403
```

## Local admin without Access

Direct loopback calls are allowed by default. To lock those down too, set
`FISHSEUS_LOCAL_TOKEN` and send it on each request:

```bash
curl -H "X-Fishseus-Token: $FISHSEUS_LOCAL_TOKEN" http://127.0.0.1:8000/api/status
```
