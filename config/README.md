# Config Module

Stores prompt and runtime configuration assets used by higher-level services.

## Files
- `personality_prompt.txt` – the fish's character and voice, loaded by `AssistantService`.
- `prompt_texts.json` – *(created on first edit)* customised response rules, follow-up and
  fallback lines. Only keys that differ from the defaults in `assistant/prompts.py` are stored.
- `tool_overrides.json` – *(created on first edit)* per-tool edits from the web UI: enabled,
  description, hint, examples. Only differences from the defaults in `assistant/tools.py`.
- `fish_config.json` – per-service settings, read by `fishseus.py` (committed; served by the web UI).
- `secrets.json` – credentials (Spotify client ID/secret, Wi-Fi passphrase). Gitignored; copy `secrets.example.json`. Read via `services.load_secrets()`.

## How it is used
- The personality, response rules, tool list and long-term memory are combined into the
  system prompt on every turn (see [assistant/README.md](../assistant/README.md)).
- All three prompt files are editable from the web UI (Personality and Tools pages) and are
  re-read when they change, so edits apply to the next message without a restart.
