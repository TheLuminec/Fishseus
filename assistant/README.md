# Assistant Module

The fish "brain" for Fishseus: owns personality, memory, and conversation
history; builds prompts for `LlmService`; parses structured model responses; and
validates + executes safe local tool calls, then turns tool results into a
spoken answer.

**Responsibilities:** personality + prompt text, long-term memory, recalled
conversation, message building, response parsing, tool execution.

**Non-responsibilities:** no audio/STT/TTS, no direct GPIO (only via tools the
orchestrator wires in), and it doesn't know which model/server is behind
`LlmService`.

| File                   | Contents                                                              |
| ---------------------- | --------------------------------------------------------------------- |
| `assistant_service.py` | `AssistantService`, `AssistantConfig`, `Tool`, `ToolRegistry`         |
| `memory.py`            | `MemoryStore` (long-term) and `ConversationStore` (recalled turns)    |
| `prompts.py`           | Default prompt text (response rules, follow-up, fallbacks) + overrides |
| `tools.py`             | Tool implementations with their hints and examples                    |

## Memory

There are two kinds, and the model sees both on every turn.

**Long-term memory** (`data/assistant_memory.json`) holds what the fish was asked
to keep: the human's name, response preferences, standing notes (written in the
web UI), and facts. Facts are one complete sentence under a stable snake_case
label (`[ethan_address] Ethan Weber lives at …`). Writing an existing label
replaces the sentence, so corrections update instead of piling up. All of it is
shown to the model; nothing is silently cut off.

- The model writes via `"memory_updates"` when asked ("remember that…", "add to
  your memory…", "call me…"). With `require_explicit_memory_intent`, new memories
  without such a request are ignored, questions like "what do you remember
  about X?" never count as a request, and only an actual name ("call me…",
  "my name is…") can change the human's name. Corrections to an existing fact are
  always allowed.
- If the model ignores an explicit request, a regex fallback stores the fact.
- `forget` deletes one fact. `remember_conversation` summarises the recalled
  conversation into a dated summary plus lasting facts (reusing existing labels).
- v1 files (with `session` state and appended duplicates) migrate automatically.

**Conversation memory** (`data/conversation_log.jsonl`) is every turn, appended as
it happens: the user's words, the model's reply, tool calls and results, the
follow-up answer, and memory writes. Turns from the last `recall_hours` (after
the latest "start fresh") are replayed into each prompt, so the fish keeps the
thread across restarts. They are replayed in the same JSON format the model must
produce, including its tool calls and their results, so the conversation itself
demonstrates correct behaviour. "Start fresh" (`clear_session` tool or web UI)
appends a marker instead of deleting anything.

## What the model sees

```
system:  personality  +  response rules  +  AVAILABLE TOOLS (with examples)  +  LONG-TERM MEMORY
         [time marker]                       ← when there's a 15+ minute gap
user / assistant ...                         ← recalled turns, replayed as JSON
system:  [Now: Saturday, September 26, 2026, 3:15 PM]
user:    the new message
```

Stable parts come first so providers can cache the prefix. The web UI's
**Memory → What the Fish Sees** shows this exact list.

Only tools that are enabled and whose service is running are listed, each with
its description, hint and examples. Edit them on the web UI's Tools page
(saved to `config/tool_overrides.json`). The personality lives in
`config/personality_prompt.txt`; everything else the model is told is in
`prompts.py`, editable on the Personality page (saved to `config/prompt_texts.json`).

## Configuration (`AssistantConfig`)

Built from the `assistant` section of `fish_config.json` via
`AssistantConfig.from_section(section, config_dir=...)`. Paths are relative to `config/`.

| Field                            | Default                                | Purpose                                              |
| -------------------------------- | -------------------------------------- | ---------------------------------------------------- |
| `assistant_name`                 | `"Fishseus"`                           | The fish's name.                                     |
| `user_name`                      | `""`                                   | Seeds long-term memory until the human says theirs.  |
| `personality_path`               | `<root>/config/personality_prompt.txt` | Character and voice.                                 |
| `prompt_texts_path`              | `<root>/config/prompt_texts.json`      | Prompt text overrides.                               |
| `memory_path`                    | `<root>/data/assistant_memory.json`    | Long-term memory.                                    |
| `history_path`                   | `<root>/data/conversation_log.jsonl`   | Conversation log / recall source.                    |
| `recall_hours`                   | `12.0`                                 | How far back the conversation is recalled (0 = all). |
| `max_recall_turns`               | `150`                                  | Cap on recalled turns (0 = no cap).                  |
| `max_tool_calls`                 | `3`                                    | Tool calls allowed per turn.                         |
| `temperature`                    | `0.7`                                  | LLM sampling temperature.                            |
| `max_tokens`                     | `350`                                  | LLM response cap.                                    |
| `require_explicit_memory_intent` | `True`                                 | Only save new memories when asked.                   |

`config.validate()` raises `AssistantServiceError` on negative counts. The recall
settings, temperature, token and tool limits can be changed live with
`apply_settings()` (the web UI does this).

## API

- `handle_user_text(text, *, source="voice", say=None) -> AssistantResult` – one
  full turn: ask the model, store memories, run tools, get the follow-up answer
  from tool results, record the turn. `say(text, motion, final)` is called for
  each line to speak: an optional remark while a tool runs (`final=False`), then
  the answer (`final=True`). The web UI passes no `say` and reads `spoken_text`.
- `handle_sensor_event(description) -> AssistantResult` – in-character reaction
  to a sensor trigger; recorded in the conversation if the fish speaks.
- `remember_conversation(focus="") -> str` – summarise the recalled conversation
  into long-term memory.
- `clear_history()` – start fresh (stop recalling the conversation so far).
- `context_preview() -> list[dict]` – the messages the model would get next.
- Lifecycle: `initialize()`, `shutdown()` (persists memory), `reset()` (reloads
  from disk), `status()`.

`AssistantResult` carries `speak` (first reply, may be `""`), `motion`,
`tool_calls`, `tool_results`, `memory_updates` (as actually stored), `answer` /
`answer_motion` (after tool results), `spoken_text` (everything said), and
`elapsed_s`.

## Usage

```python
assistant = AssistantService(llm=llm, config=AssistantConfig(), tool_registry=tools)
assistant.initialize()
result = assistant.handle_user_text("what's 47 times 83?",
                                    say=lambda text, motion, final: print(text))
assistant.shutdown()   # persists memory
```

## Requirements

- An initialized `LlmService`, a personality prompt file, and writable `data/`
  for memory + the conversation log.
