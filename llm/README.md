# LLM Module

Thin OpenAI-compatible chat client service for Fishseus. Sends
`/v1/chat/completions` requests to a local or remote endpoint (Ollama, xAI Grok,
OpenAI, LM Studio, …) and returns parsed content. Intentionally backend-only.

**Responsibilities:** build/send chat requests, handle timeouts + retries, parse
responses.

**Non-responsibilities:** no memory, personality, tool policy, or orchestration —
those belong to `assistant_service.py`.

## Configuration (`LlmConfig`)

| Field               | Default                          | Purpose                                            |
| ------------------- | -------------------------------- | -------------------------------------------------- |
| `module_name`       | `"llm"`                          | Service key in the orchestrator config.            |
| `provider`          | `""`                             | Name of the provider this was resolved from.       |
| `api_type`          | `"ollama"`                       | `ollama` or `openai` request style (see below).    |
| `endpoint_url`      | Ollama `…/v1/chat/completions`   | Chat-completions endpoint.                         |
| `model`             | `"qwen2.5:3b"`                   | Model name the server expects.                     |
| `api_key`           | `None`                           | Optional bearer token.                             |
| `timeout_s`         | `45.0`                           | Per-request timeout.                               |
| `retries`           | `1`                              | Extra attempts on timeout/connection error.        |
| `retry_delay_s`     | `0.5`                            | Delay between retries.                             |
| `temperature`       | `0.7`                            | Sampling temperature.                             |
| `max_tokens`        | `250`                            | Response cap.                                      |
| `top_p`             | `None`                           | Optional nucleus sampling.                        |
| `disable_reasoning` | `True`                           | `ollama` only: ask thinking models to skip thinking. |
| `reasoning_effort`  | `None`                           | `openai` only: sent when set (e.g. `"low"`).       |
| `extra_payload`     | `{}`                             | Extra fields merged into the request body.         |

`config.validate()` raises `LlmServiceError` if `endpoint_url` or `model` is
empty, `timeout_s <= 0`, or `api_type` is unknown. It runs in `initialize()`.

## Providers

The `"llm"` section of `fish_config.json` holds named providers plus the shared
generation settings; `provider` picks the active one:

```json
"llm": {
  "provider": "xai",
  "temperature": 0.75,
  "max_tokens": 1024,
  "providers": {
    "ollama": {"label": "Ollama (tailnet)", "api_type": "ollama",
               "endpoint_url": "http://ollama.angelfish-gamma.ts.net/v1/chat/completions",
               "model": "qwen2.5:3b", "disable_reasoning": false},
    "xai":    {"label": "xAI Grok", "api_type": "openai",
               "endpoint_url": "https://api.x.ai/v1/chat/completions",
               "model": "grok-4.20-0309-non-reasoning", "api_key_env": "XAI_API_KEY"}
  }
}
```

`config_from_section(section)` lays the active entry over the shared settings and
returns one flat `LlmConfig`. A section without `providers` is read as a flat
`LlmConfig` (the old format). The API key is resolved in order: the entry's
`api_key` → `config/secrets.json` `"llm_api_keys": {"<provider>": "..."}` → the
env var named by `api_key_env`. Keys never go in `fish_config.json`, which is
committed and served by the web UI.

**Request styles (`api_type`):**

- `ollama` – sends `max_tokens`, and with `disable_reasoning` also Ollama's
  thinking switches (`think: false`, `reasoning_effort: "none"`).
- `openai` – strict OpenAI parameters only: `max_completion_tokens`, plus
  `reasoning_effort` when configured. Strict servers reject Ollama's extras (xAI
  refuses `reasoning_effort: "none"`).

**xAI Grok notes:** use `grok-4.20-0309-non-reasoning` for the fish: it answers
without a thinking pass, so replies are fast and the assistant's small token
budgets (150–350) go to the answer. Reasoning models such as `grok-4.7` spend
part of that budget thinking (if nothing is left for the answer, you get
`LLM returned reasoning but no final content`); if you use one, set
`reasoning_effort` to `"low"`. xAI marks `/v1/chat/completions` as legacy in
favour of its Responses API, but it still works and no sunset date has been
announced.

The web UI's **Language Model** page lists providers, stores keys (write-only),
loads a provider's model list (`GET <base>/models`), runs a test request in the
assistant's JSON mode with timing, and switches the live provider without a
restart.

## Lifecycle API

- `initialize()` – validate config only (no network call, so a down server never
  blocks start-up). Use `health_check()` to actually probe the endpoint.
- `shutdown()` – close the HTTP session.
- `reset()` – `shutdown()` then `initialize()`.
- `status()` – `{enabled, service, provider, model, endpoint}`.

## Chat API

- `chat(messages, *, model=…, temperature=…, max_tokens=…, tools=…, …) -> LlmResult`
  – send a chat completion; pass-through kwargs for OpenAI-style tool calling.
- `health_check() -> bool` – tiny live request that confirms the endpoint works.
- `list_models() -> list[str]` – model ids from the endpoint's OpenAI-style
  `GET /models`; raises `LlmServiceError` on failure.

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
