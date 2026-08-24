# Assistant Module

The fish "brain" for Fishseus: owns personality, memory, and conversation
history; builds prompts for `LlmService`; parses structured model responses; and
validates + executes safe local tool calls.

**Responsibilities:** personality prompt, short-term history, JSON-backed memory,
message building, response parsing, tool execution.

**Non-responsibilities:** no audio/STT/TTS, no direct GPIO (only via tools the
orchestrator wires in), and it doesn't know which model/server is behind
`LlmService`.

## Configuration (`AssistantConfig`)

| Field                            | Default                              | Purpose                                       |
| -------------------------------- | ------------------------------------ | --------------------------------------------- |
| `module_name`                    | `"assistant"`                        | Service key in the orchestrator config.       |
| `assistant_name`                 | `"Fishseus"`                         | The fish's name.                              |
| `user_name`                      | `"Caleb"`                            | Default user name.                            |
| `personality_path`               | `<root>/config/personality_prompt.txt` | Personality prompt file.                    |
| `memory_path`                    | `<root>/data/assistant_memory.json`  | Persistent memory store.                      |
| `history_path`                   | `<root>/data/conversation_log.jsonl` | Conversation log.                             |
| `max_history_turns`              | `8`                                  | Turns kept in prompt context.                 |
| `max_tool_calls`                 | `3`                                  | Tool calls allowed per turn.                  |
| `temperature`                    | `0.7`                                | LLM sampling temperature.                     |
| `max_tokens`                     | `350`                                | LLM response cap.                             |
| `require_explicit_memory_intent` | `True`                               | Only save memory when the user asks to.       |

`config.validate()` raises `AssistantServiceError` on negative
`max_history_turns` / `max_tool_calls`. It runs in `initialize()`.

## Lifecycle API

- `initialize()` – validate config and (re)load memory + personality. `__init__`
  already loads them, so this is idempotent.
- `shutdown()` – persist memory to disk.
- `reset()` – clear history and reload memory.
- `status()` – `{enabled, service, history_turns, tools}`.

## Assistant API

- `handle_user_text(text) -> AssistantResult` – main entry point for a user
  utterance (wake word already stripped upstream).
- `handle_sensor_event(description) -> AssistantResult` – let the fish decide how
  to react to a sensor trigger.
- `formulate_tool_response(command, tool_results) -> (speak, motion)` – fold tool
  output into a single natural spoken reply and its matching motion.
- `set_last_response(text)` – overwrite the latest assistant history entry with the
  text actually spoken, so silent tool calls and formulated answers stay in history.
- `clear_history()` – drop in-memory conversation history.

`AssistantResult` carries `speak`, `motion`, `tool_calls`, `tool_results`,
`memory_updates`, and `elapsed_s`. The orchestrator sends `speak` to TTS and uses
`motion`/tool results to drive animation. `speak` may be an empty string when the
fish acts silently (a bare action, or a tool call whose result is spoken afterward).

## How it works

- Loads personality + persistent memory, builds a message list from recent
  history, and asks `LlmService` for a JSON response.
- Parses the response into speech, a motion hint, tool calls, and memory updates;
  validates and executes tools via the `ToolRegistry` (respecting `max_tool_calls`
  and each tool's risk level).
- Appends the turn to the history log; memory writes honor
  `require_explicit_memory_intent`.

## Usage

```python
assistant = AssistantService(llm=llm, config=AssistantConfig(), tool_registry=tools)
assistant.initialize()
result = assistant.handle_user_text("wiggle twice")
print(result.speak, result.motion)
assistant.shutdown()   # persists memory
```

## Requirements

- An initialized `LlmService`, a personality prompt file, and writable `data/`
  for memory + history.
