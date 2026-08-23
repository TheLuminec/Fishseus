# LLM Module

Thin OpenAI-compatible chat client service for Fishseus. Sends
`/v1/chat/completions` requests to a local or remote endpoint (Ollama, etc.) and
returns parsed content. Intentionally backend-only.

**Responsibilities:** build/send chat requests, handle timeouts + retries, parse
responses.

**Non-responsibilities:** no memory, personality, tool policy, or orchestration —
those belong to `assistant_service.py`.

## Configuration (`LlmConfig`)

| Field               | Default                          | Purpose                                            |
| ------------------- | -------------------------------- | -------------------------------------------------- |
| `module_name`       | `"llm"`                          | Service key in the orchestrator config.            |
| `endpoint_url`      | Ollama `…/v1/chat/completions`   | Chat-completions endpoint.                         |
| `model`             | `"qwen2.5:3b"`                   | Model name the server expects.                     |
| `api_key`           | `None`                           | Optional bearer token.                             |
| `timeout_s`         | `45.0`                           | Per-request timeout.                               |
| `retries`           | `1`                              | Extra attempts on timeout/connection error.        |
| `retry_delay_s`     | `0.5`                            | Delay between retries.                             |
| `temperature`       | `0.7`                            | Sampling temperature.                             |
| `max_tokens`        | `250`                            | Response cap.                                      |
| `top_p`             | `None`                           | Optional nucleus sampling.                        |
| `disable_reasoning` | `True`                           | Ask thinking models for direct final answers.      |
| `extra_payload`     | `{}`                             | Extra fields merged into the request body.         |

`config.validate()` raises `LlmServiceError` if `endpoint_url` is empty or
`timeout_s <= 0`. It runs in `initialize()`.

## Lifecycle API

- `initialize()` – validate config only (no network call, so a down server never
  blocks start-up). Use `health_check()` to actually probe the endpoint.
- `shutdown()` – close the HTTP session.
- `reset()` – `shutdown()` then `initialize()`.
- `status()` – `{enabled, service, model, endpoint}`.

## Chat API

- `chat(messages, *, model=…, temperature=…, max_tokens=…, tools=…, …) -> LlmResult`
  – send a chat completion; pass-through kwargs for OpenAI-style tool calling.
- `health_check() -> bool` – tiny live request that confirms the endpoint works.

`LlmResult` carries `content`, `elapsed_s`, `model`, and `raw_response`, plus
`parse_json_content()` for tolerant JSON extraction (handles fenced/messy output).

## How it works

- Builds a JSON payload from `LlmConfig` + per-call overrides (including
  reasoning/thinking toggles for Ollama-compatible servers).
- Posts to the endpoint with timeout + retry on `Timeout`/`RequestException`.
- Extracts `choices[0].message.content`, raising `LlmResponseError` on an
  unexpected shape or a reasoning-only response.

## Usage

```python
llm = LlmService(LlmConfig(model="qwen2.5:3b"))
llm.initialize()
result = llm.chat([{"role": "user", "content": "Say hi in one sentence."}])
print(result.content)
llm.shutdown()
```

## Requirements

- A reachable OpenAI-compatible `/v1/chat/completions` endpoint, and `requests`.
