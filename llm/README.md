# LLM Module

Provides an OpenAI-compatible chat-completions client for local/remote model endpoints.

## Main API (`llm_service.py`)
- `LlmService.chat(messages, **overrides)` – sends chat messages and returns `LlmResponse`.
- `LlmService.health_check()` – basic connectivity check to endpoint.
- `LlmService.close()` – closes the underlying HTTP session.

## How it works
- Builds JSON payload from `LlmConfig` + per-call overrides.
- Sends HTTP request to `/v1/chat/completions` style endpoint.
- Applies timeout/retry behavior.
- Returns normalized assistant content plus raw metadata on `LlmResponse`.

## Typical usage
```python
llm = LlmService(LlmConfig(endpoint_url="http://host/v1/chat/completions", model="gemma"))
response = llm.chat([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Say hi."},
])
print(response.content)
```
