"""
llm_service.py

Thin OpenAI-compatible LLM client service for Fishseus.

Responsibilities:
- Send chat/completions requests to an OpenAI-compatible endpoint (Ollama,
  xAI Grok, OpenAI, ...)
- Handle timeouts, retries, and basic response parsing
- Resolve named providers from config (config_from_section) so the endpoint
  can be switched without code changes

Non-responsibilities:
- No memory, personality, tool policy, or orchestration — those belong to
  assistant_service.py

Expected API shape:
    POST /v1/chat/completions
    {"model": "...", "messages": [{"role": "user", "content": "hello"}],
     "temperature": 0.7, "max_tokens": 250}
    -> {"choices": [{"message": {"role": "assistant", "content": "..."}}]}

Example:
    llm = LlmService()
    llm.initialize()
    result = llm.chat([{"role": "user", "content": "Introduce yourself briefly."}])
    print(result.content)
    llm.shutdown()

Provider config ("llm" in fish_config.json):
    {"provider": "xai", "temperature": 0.75,
     "providers": {
        "ollama": {"api_type": "ollama", "endpoint_url": "http://.../v1/chat/completions",
                   "model": "qwen2.5:3b"},
        "xai":    {"api_type": "openai", "endpoint_url": "https://api.x.ai/v1/chat/completions",
                   "model": "grok-4.20-0309-non-reasoning"}}}
    API keys live in config/secrets.json under "llm_api_keys": {"xai": "xai-..."}.

Orchestrator usage:
    llm = LlmService(config_from_section(config.get("llm", {})))
    llm.initialize()                 # validates config (no network call)
    assistant = AssistantService(llm=llm, ...)
    ...
    llm.shutdown()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, fields
from typing import Any, Optional

import requests

from services import Service, ServiceConfig, ServiceError, load_secrets

# api_type values:
#   "ollama" - Ollama's OpenAI-compatible endpoint; also sends Ollama's thinking
#              switches (think / reasoning_effort "none") when disable_reasoning.
#   "openai" - strict OpenAI-style params only (xAI, OpenAI, LM Studio, ...).
#              Strict servers reject Ollama's extras, e.g. xAI refuses
#              reasoning_effort "none".
API_TYPES = ("ollama", "openai")
# Secrets section holding one API key per provider name.
SECRETS_SECTION = "llm_api_keys"


class LlmServiceError(ServiceError):
    """Base error for LLM service failures."""


class LlmTimeoutError(LlmServiceError):
    """Raised when the LLM request repeatedly times out."""


class LlmResponseError(LlmServiceError):
    """Raised when the LLM server returns an invalid or unexpected response."""


@dataclass(frozen=True)
class LlmConfig(ServiceConfig):
    """
    Configuration for an OpenAI-compatible chat completion endpoint.

    endpoint_url should point directly to /v1/chat/completions.
    """

    module_name: str = "llm"

    # Display name of the provider this config was resolved from (status only).
    provider: str = ""
    api_type: str = "ollama"

    endpoint_url: str = "http://ollama.angelfish-gamma.ts.net/v1/chat/completions"

    # Change this to whatever model name your Ollama/OpenAI-compatible server expects.
    # Examples:
    #   "llama3.2:3b"
    #   "qwen2.5:3b"
    #   "mistral"
    #   "devstral"
    model: str = "qwen2.5:3b"

    api_key: Optional[str] = None

    timeout_s: float = 45.0
    retries: int = 1
    retry_delay_s: float = 0.5

    temperature: float = 0.7
    max_tokens: int = 250
    top_p: Optional[float] = None

    # Thinking/reasoning controls.
    # Ollama thinking-capable models may return reasoning separately from content.
    # For a voice assistant, we usually want direct final answers only.
    disable_reasoning: bool = True
    # api_type "openai" only: sent as reasoning_effort when set (e.g. "low" for
    # grok-4.7). Leave empty for non-reasoning models.
    reasoning_effort: Optional[str] = None

    # Some local servers accept extra fields; keep them configurable.
    extra_payload: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        if not self.endpoint_url:
            raise LlmServiceError("endpoint_url must be set")
        if self.timeout_s <= 0:
            raise LlmServiceError(f"timeout_s must be positive: {self.timeout_s}")
        if self.api_type not in API_TYPES:
            raise LlmServiceError(f"api_type must be one of {API_TYPES}: {self.api_type!r}")
        if not self.model:
            raise LlmServiceError("model must be set")
        return True


def config_from_section(section: dict[str, Any]) -> LlmConfig:
    """
    Build an LlmConfig from the "llm" config section.

    With a "providers" map, the entry named by "provider" is laid over the
    shared settings (temperature, max_tokens, ...). Without one, the section is
    a flat LlmConfig (the pre-provider format). The API key comes from the
    entry's "api_key", else config/secrets.json "llm_api_keys"[provider], else
    the env var named by "api_key_env".
    """
    section = dict(section)
    providers = section.pop("providers", None) or {}
    name = str(section.pop("provider", "") or "")
    if providers:
        if name not in providers:
            raise LlmServiceError(
                f"unknown LLM provider {name!r}; configured: {sorted(providers)}")
        section.update(providers[name])
    section.pop("label", None)
    key_env = section.pop("api_key_env", None)
    if not section.get("api_key"):
        key = load_secrets(SECRETS_SECTION).get(name) if name else None
        if not key and key_env:
            key = os.environ.get(key_env)
        section["api_key"] = key or None
    section["provider"] = name
    known = {f.name for f in fields(LlmConfig)}
    unknown = set(section) - known
    if unknown:
        raise LlmServiceError(f"unknown LLM setting(s) {sorted(unknown)} for provider {name!r}")
    return LlmConfig(**section)


def models_url(endpoint_url: str) -> str:
    """The OpenAI-style model-list URL next to a chat/completions endpoint."""
    base = endpoint_url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base + "/models"


@dataclass(frozen=True)
class LlmMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LlmResult:
    content: str
    elapsed_s: float
    model: str
    raw_response: dict[str, Any]

    def parse_json_content(self) -> Optional[dict[str, Any]]:
        """
        Best-effort JSON parser for assistant_service.

        The assistant layer can ask the model to respond with JSON. Local models
        sometimes wrap JSON in markdown fences or add small amounts of text, so
        this method tries a few safe extraction strategies.
        """
        text = self.content.strip()

        if not text:
            return None

        # Direct JSON.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Markdown fenced JSON.
        if "```" in text:
            extracted = _extract_markdown_json_block(text)
            if extracted:
                try:
                    parsed = json.loads(extracted)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

        # First {...} object in a messy response.
        extracted = _extract_first_json_object(text)
        if extracted:
            try:
                parsed = json.loads(extracted)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None


class LlmService(Service):
    """
    Thin OpenAI-compatible chat client.

    This class should remain backend-focused. Do not add fish memory, prompts,
    tool execution, GPIO, or TTS here.
    """

    def __init__(self, config: LlmConfig = LlmConfig()) -> None:
        self.config = config
        self._session: Optional[requests.Session] = requests.Session()
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        # Validation only — no network call, so a down server doesn't block
        # start-up.  Use health_check() explicitly to probe the endpoint.
        self.config.validate()
        if self._session is None:
            self._session = requests.Session()
        self._initialized = True

    def shutdown(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._initialized = False

    def reset(self) -> bool:
        self.shutdown()
        self.initialize()
        return True

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "service": "ok" if self._initialized else "uninitialized",
            "provider": self.config.provider,
            "model": self.config.model,
            "endpoint": self.config.endpoint_url,
        }

    def list_models(self) -> list[str]:
        """Model ids the endpoint advertises (GET <base>/models). Raises on failure."""
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        session = self._session or requests
        try:
            response = session.get(models_url(self.config.endpoint_url),
                                   headers=headers, timeout=min(self.config.timeout_s, 10))
        except requests.RequestException as exc:
            raise LlmServiceError(f"model list request failed: {exc}") from exc
        if response.status_code >= 400:
            raise LlmResponseError(
                f"model list returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = response.json().get("data") or []
        except (ValueError, AttributeError) as exc:
            raise LlmResponseError("model list was not OpenAI-style JSON") from exc
        return sorted(str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id"))

    def chat(
        self,
        messages: list[dict[str, str]] | list[LlmMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[dict[str, Any]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> LlmResult:
        """
        Send a chat completion request and return assistant content.

        Parameters like tools/tool_choice are passed through if your model server
        supports OpenAI-style tool calling. assistant_service.py should still own
        whether tools are allowed and how they execute.
        """
        payload = self._build_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            extra_payload=extra_payload,
        )

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        attempts = max(1, self.config.retries + 1)
        last_error: Optional[BaseException] = None
        start = time.monotonic()
        session = self._session or requests

        for attempt in range(attempts):
            try:
                response = session.post(
                    self.config.endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_s,
                )

                if response.status_code >= 400:
                    raise LlmResponseError(
                        f"LLM server returned HTTP {response.status_code}: {response.text[:1000]}"
                    )

                raw = response.json()
                content = self._extract_content(raw)
                elapsed = time.monotonic() - start

                return LlmResult(
                    content=content,
                    elapsed_s=elapsed,
                    model=payload["model"],
                    raw_response=raw,
                )

            except requests.Timeout as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(self.config.retry_delay_s)
                    continue
                raise LlmTimeoutError(
                    f"LLM request timed out after {attempts} attempt(s)"
                ) from exc

            except requests.RequestException as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(self.config.retry_delay_s)
                    continue
                raise LlmServiceError(f"LLM request failed: {exc}") from exc

            except json.JSONDecodeError as exc:
                last_error = exc
                raise LlmResponseError("LLM server returned non-JSON response") from exc

        raise LlmServiceError(f"LLM request failed: {last_error}")

    def health_check(self) -> bool:
        """
        Lightweight practical health check using a tiny chat request.

        Some OpenAI-compatible local servers do not expose /models reliably, so
        this confirms the actual endpoint works.
        """
        try:
            result = self.chat(
                [
                    {"role": "system", "content": "Reply with only: ok"},
                    {"role": "user", "content": "ping"},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            return bool(result.content.strip())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_payload(
        self,
        *,
        messages: list[dict[str, str]] | list[LlmMessage],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float],
        response_format: Optional[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        tool_choice: Optional[str | dict[str, Any]],
        extra_payload: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, LlmMessage):
                normalized_messages.append(msg.to_dict())
            else:
                normalized_messages.append({"role": msg["role"], "content": msg["content"]})

        effective_max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": normalized_messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }

        if self.config.api_type == "ollama":
            payload["max_tokens"] = effective_max_tokens
            # Ollama thinking controls. Different Ollama versions/models have used
            # different knobs, so include both; Ollama ignores the one it lacks.
            if self.config.disable_reasoning:
                payload["reasoning_effort"] = "none"
                payload["think"] = False
        else:
            # Strict OpenAI-style servers: max_tokens is deprecated there (xAI,
            # OpenAI), and unknown or out-of-range fields are rejected.
            payload["max_completion_tokens"] = effective_max_tokens
            if self.config.reasoning_effort:
                payload["reasoning_effort"] = self.config.reasoning_effort

        effective_top_p = self.config.top_p if top_p is None else top_p
        if effective_top_p is not None:
            payload["top_p"] = effective_top_p

        if response_format is not None:
            payload["response_format"] = response_format

        if tools is not None:
            payload["tools"] = tools

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        payload.update(self.config.extra_payload)
        if extra_payload:
            payload.update(extra_payload)

        return payload

    @staticmethod
    def _extract_content(raw: dict[str, Any]) -> str:
        try:
            choice = raw["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")

            if content is None:
                # Some servers may provide old completions-style text.
                content = choice.get("text")

            if content is None or content == "":
                # Tool-only or reasoning-only responses may not include normal
                # content. Do not use reasoning as speakable output, but make
                # the failure easier to diagnose.
                reasoning = message.get("reasoning") or message.get("reasoning_content")
                if reasoning:
                    raise LlmResponseError(
                        "LLM returned reasoning but no final content. "
                        "Try disabling reasoning/thinking for this model or raising max_tokens."
                    )
                return ""

            if not isinstance(content, str):
                raise TypeError(f"message.content was not a string: {type(content)}")

            return content.strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmResponseError(f"Unexpected LLM response shape: {raw}") from exc


def _extract_markdown_json_block(text: str) -> Optional[str]:
    parts = text.split("```")
    for part in parts:
        candidate = part.strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    return None


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract the first balanced JSON-looking object from text.
    This avoids brittle regex for nested braces.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None
