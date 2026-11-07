# claude_link

Two-way link between the fish and Claude Code on the Pi.

## Fish → Claude
The `ask_claude` tool (in `assistant/tools.py`) calls `ClaudeLinkService.ask()`. That starts a headless
`claude -p` run in a background thread named **Fish Liaison**, in `~/Docs/fish/liaison/`. The fish says
"Okay, I'm asking Claude…" right away and keeps listening. When the reply arrives, it goes into the
orchestrator's announce queue, and the fish speaks it on the next pass of the main loop. Expect a delay
of up to `audio.max_record_seconds`.

The liaison can only read files, search the web, and use `ListAgents` and `SendMessage`. It answers
directly, or relays to the **Fishseus Agent** session (or any session it can see). Only one request
runs at a time.

## Claude → Fish
Any process can make the fish speak by dropping a `.txt` file into `~/Docs/fish/say/`:

```bash
bin/fishsay "Dinner is ready"
```

Files older than `outbox_max_age_s` (default 10 min) are skipped, so messages sent while the fish
was off aren't blurted out later.

## Log
Every exchange, and every outbox line, is appended to `~/Docs/fish/claude_link.jsonl`. The log records
the timestamp, message, reply, cost, elapsed time, and Claude session id. The last `history_turns`
entries are given to each new liaison run as context.

## Config (`claude_link` section of `fish_config.json`)
| key | default | |
|---|---|---|
| `enabled` | true | |
| `model` | "" | `""` = account default; e.g. `"sonnet"` for lower latency |
| `timeout_s` | 180 | per request |
| `history_turns` | 6 | context carried between runs |
| `allowed_tools` | read/search/messaging | space-separated Claude Code tool names |

Smoke test (no audio): `python3 -m claude_link.claude_link_service "what's two plus two?"`
