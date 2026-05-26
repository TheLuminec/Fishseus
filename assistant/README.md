# Assistant Module

Implements assistant behavior: prompt assembly, memory/history persistence, LLM calls, and optional tool execution.

## Main API (`assistant_service.py`)
- `AssistantService.handle_user_text(text)` – primary entry point for a user utterance.
- `ToolRegistry.register(tool)` – registers callable tools exposed to the assistant.
- `ToolRegistry.execute(name, args)` – executes a tool and returns tool output.

## How it works
- Loads personality prompt and persistent memory.
- Builds conversation context from recent history.
- Sends request through `LlmService`.
- Parses and executes tool calls when present.
- Appends interactions to history/memory files.

## Typical usage
```python
assistant = AssistantService(llm=llm, config=AssistantConfig(...), tool_registry=tools)
response = assistant.handle_user_text("fish, wiggle twice")
print(response.text)
```
