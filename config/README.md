# Config Module

Stores prompt and runtime configuration assets used by higher-level services.

## Files
- `personality_prompt.txt` – system personality/instruction prompt loaded by `AssistantService`.
- `fish_config.json` – per-service settings, read by `fishseus.py` (committed; served by the web UI).
- `secrets.json` – credentials (Spotify client ID/secret, Wi-Fi passphrase). Gitignored; copy `secrets.example.json`. Read via `services.load_secrets()`.

## How it is used
- `AssistantService` reads this file at startup and injects it into LLM system context.
- Adjusting this file changes assistant tone/behavior without code changes.
