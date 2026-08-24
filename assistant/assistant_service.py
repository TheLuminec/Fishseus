"""
assistant_service.py

Demo assistant brain for the Fishseus/Billy Bass assistant project.

Responsibilities:
- Own the fish personality prompt.
- Own short-term conversation history.
- Own JSON-backed working/persistent memory.
- Build messages for llm_service.py.
- Parse structured model responses.
- Validate and execute safe local tool calls.

Non-responsibilities:
- Does not record audio.
- Does not run Whisper/STT.
- Does not play TTS.
- Does not directly own GPIO unless tools are wired in by the orchestrator.
- Does not know what model/server is being used beyond the LlmService interface.

Expected flow:
    assistant = AssistantService(llm=llm)
    result = assistant.handle_user_text("wiggle twice")
    print(result.speak)
    print(result.motion)

The orchestrator should then:
    - send result.speak to TTS
    - play the TTS audio
    - pass the WAV to motion_service.speak_audio(...)
    - optionally use result.motion to trigger extra animation
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from llm.llm_service import LlmService, LlmServiceError
from services import Service, ServiceConfig, ServiceError, ROOT_DIR


class AssistantServiceError(ServiceError):
    pass


# Motions the model may request. Anything outside this set falls back to "speaking".
VALID_MOTIONS = {"idle", "speaking", "happy", "annoyed", "thinking", "excited"}


# ----------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class AssistantConfig(ServiceConfig):
    module_name: str = "assistant"

    assistant_name: str = "Fishseus"
    user_name: str = "Caleb"

    personality_path: Path = ROOT_DIR / "config" / "personality_prompt.txt"
    memory_path: Path = ROOT_DIR / "data" / "assistant_memory.json"
    history_path: Path = ROOT_DIR / "data" / "conversation_log.jsonl"

    max_history_turns: int = 8
    max_tool_calls: int = 3

    temperature: float = 0.7
    max_tokens: int = 350

    # If true, memory is only saved when the user's command explicitly asks
    # the assistant to remember something.
    require_explicit_memory_intent: bool = True

    def validate(self) -> bool:
        if self.max_history_turns < 0:
            raise AssistantServiceError(
                f"max_history_turns must be >= 0: {self.max_history_turns}"
            )
        if self.max_tool_calls < 0:
            raise AssistantServiceError(
                f"max_tool_calls must be >= 0: {self.max_tool_calls}"
            )
        return True


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantResult:
    speak: str
    motion: str = "speaking"
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    memory_updates: list[dict[str, Any]] = field(default_factory=list)
    raw_model_text: str = ""
    parsed_json: Optional[dict[str, Any]] = None
    elapsed_s: float = 0.0


@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., Any]
    risk: str = "safe"              # safe | confirm | blocked
    returns_data: bool = True       # False for fire-and-forget actions (wiggle, open_mouth)
    synthesize_result: bool = False # True = feed result back to LLM for a natural spoken response
                                    # False = speak the raw result directly (good for short values)
    enabled: bool = True            # False = hidden from LLM prompt and cannot be executed


# ----------------------------------------------------------------------
# Memory store
# ----------------------------------------------------------------------

class MemoryStore:
    """
    Small JSON-backed memory store.

    This starts intentionally simple. Later, you can replace this with SQLite,
    embeddings, a remote memory service, or a richer profile system without
    changing the LLM transport layer.
    """

    def __init__(self, path: Path, assistant_name: str, user_name: str) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = self._default_memory(assistant_name, user_name)

    def load(self) -> None:
        if not self.path.exists():
            self.save()
            return

        try:
            loaded = json.loads(self.path.read_text())
            if isinstance(loaded, dict):
                self.data = self._merge_defaults(self.data, loaded)
        except Exception as exc:
            print(f"[MemoryStore] Failed to load memory, using defaults: {exc}")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def compact_summary(self) -> str:
        profile = self.data.get("profile", {})
        preferences = self.data.get("preferences", {})
        session = self.data.get("session", {})
        facts = self.data.get("facts", [])

        lines = [
            f"Assistant name: {profile.get('assistant_name', 'Fishseus')}",
            f"User name: {profile.get('user_name', 'the user')}",
            f"Current mode: {session.get('current_mode', 'assistant')}",
            f"Response style: {preferences.get('response_style', 'brief, helpful, theatrical')}",
            f"Humor level: {preferences.get('humor_level', 'medium')}",
        ]

        if isinstance(facts, list):
            for fact in facts[-10:]:
                key = fact.get("key", "fact")
                value = fact.get("value", "")
                if value:
                    lines.append(f"Remembered {key}: {value}")

        return "\n".join(f"- {line}" for line in lines)

    def update_path(self, dotted_key: str, value: Any) -> None:
        """
        Update a dotted key like "preferences.response_style".
        Unknown top-level keys are allowed but kept simple.
        """
        parts = [p for p in dotted_key.split(".") if p]
        if not parts:
            return

        node = self.data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def add_fact(self, key: str, value: str) -> None:
        facts = self.data.setdefault("facts", [])
        if not isinstance(facts, list):
            self.data["facts"] = []
            facts = self.data["facts"]

        facts.append(
            {
                "key": key,
                "value": value,
                "created_at": int(time.time()),
            }
        )

    # Words ignored when matching a "forget" request against stored facts.
    _FORGET_STOPWORDS = {
        "the", "a", "an", "my", "your", "that", "this", "about", "of", "for",
        "to", "is", "and", "please", "fishseus", "forget", "delete", "remove",
        "erase", "remember", "memory", "everything",
    }

    def forget_fact(self, query: str) -> list[dict[str, Any]]:
        """
        Remove remembered facts matching ``query`` and return the ones removed.

        Matching is deliberately forgiving because the query arrives via speech:
        an exact key match wins; otherwise every meaningful word in the query
        must appear somewhere in a fact's key or value. Returns [] (and changes
        nothing) when nothing matches. Callers persist via save().
        """
        facts = self.data.get("facts", [])
        if not isinstance(facts, list) or not facts:
            return []

        q = query.strip().lower()
        if q.startswith("facts."):
            q = q[len("facts."):]
        q = q.strip()
        if not q:
            return []

        def haystack(fact: dict[str, Any]) -> str:
            text = f"{fact.get('key', '')} {fact.get('value', '')}".lower()
            return text.replace("_", " ")

        exact = [f for f in facts if str(f.get("key", "")).lower() == q]
        if exact:
            matches = exact
        else:
            tokens = [
                t for t in re.split(r"[^a-z0-9]+", q)
                if len(t) >= 2 and t not in self._FORGET_STOPWORDS
            ]
            if tokens:
                matches = [f for f in facts if all(t in haystack(f) for t in tokens)]
            else:
                matches = [f for f in facts if q in haystack(f)]

        if not matches:
            return []

        remove_ids = {id(f) for f in matches}
        self.data["facts"] = [f for f in facts if id(f) not in remove_ids]
        return matches

    @staticmethod
    def _default_memory(assistant_name: str, user_name: str) -> dict[str, Any]:
        return {
            "profile": {
                "assistant_name": assistant_name,
                "user_name": user_name,
            },
            "preferences": {
                "response_style": "brief, helpful, technical when needed",
                "humor_level": "medium",
                "personality": "dramatic sarcastic fish oracle",
            },
            "session": {
                "current_mode": "assistant",
                "last_topic": None,
                "last_command": None,
            },
            "facts": [
                {
                    "key": "project",
                    "value": "The user is building a Raspberry Pi 5 powered Billy Bass assistant named Fishseus.",
                    "created_at": int(time.time()),
                }
            ],
        }

    @staticmethod
    def _merge_defaults(default: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
        merged = dict(default)
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = MemoryStore._merge_defaults(merged[key], value)
            else:
                merged[key] = value
        return merged


# ----------------------------------------------------------------------
# Tool registry
# ----------------------------------------------------------------------

class ToolRegistry:
    """
    Stores safe callable tools.

    The model can request tool calls, but this registry decides what actually
    executes. The LLM never directly controls hardware.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def describe_for_prompt(self) -> str:
        lines = [
            f"- {t.name}: {t.description}"
            for t in self._tools.values()
            if t.risk == "safe" and t.enabled
        ]
        return "\n".join(lines) if lines else "No tools are currently available."

    def execute(self, call: ToolCall) -> dict[str, Any]:
        tool = self._tools.get(call.name)
        if tool is None:
            return {"tool": call.name, "ok": False, "error": "Unknown tool"}

        if not tool.enabled:
            return {"tool": call.name, "ok": False, "error": "Tool is currently disabled"}

        if tool.risk != "safe":
            return {
                "tool": call.name,
                "ok": False,
                "error": f"Tool risk '{tool.risk}' is not executable in demo mode",
            }

        try:
            result = tool.function(**call.args)
            return {
                "tool": call.name,
                "ok": True,
                "result": result,
                "returns_data": tool.returns_data,
                "synthesize_result": tool.synthesize_result,
            }
        except Exception as exc:
            return {"tool": call.name, "ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Management helpers (used by web API)
    # ------------------------------------------------------------------

    def enable(self, name: str) -> None:
        if name in self._tools:
            self._tools[name].enabled = True

    def disable(self, name: str) -> None:
        if name in self._tools:
            self._tools[name].enabled = False

    def update_description(self, name: str, description: str) -> None:
        if name in self._tools:
            self._tools[name].description = description

    def set_synthesize_result(self, name: str, value: bool) -> None:
        if name in self._tools:
            self._tools[name].synthesize_result = value

    def list_all(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "risk": t.risk,
                "enabled": t.enabled,
                "returns_data": t.returns_data,
                "synthesize_result": t.synthesize_result,
            }
            for t in self._tools.values()
        ]


# ----------------------------------------------------------------------
# Assistant service
# ----------------------------------------------------------------------

class AssistantService(Service):
    def __init__(
        self,
        llm: LlmService,
        config: AssistantConfig = AssistantConfig(),
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.memory = MemoryStore(config.memory_path, config.assistant_name, config.user_name)
        self.memory.load()

        self.personality_prompt = self._load_personality_prompt(config.personality_path)

        self.tool_registry = tool_registry or ToolRegistry()
        self.history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        # __init__ already loads memory/personality; re-run here (idempotent)
        # so the orchestrator can drive every service the same way.
        self.config.validate()
        self.memory.load()
        self.personality_prompt = self._load_personality_prompt(self.config.personality_path)

    def shutdown(self) -> None:
        try:
            self.memory.save()
        except Exception as exc:
            print(f"[AssistantService] memory save failed: {exc}")

    def reset(self) -> bool:
        self.clear_history()
        self.memory.load()
        return True

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "service": "ok",
            "history_turns": len(self.history),
            "tools": len(self.tool_registry.list_all()),
        }

    def handle_user_text(self, user_text: str) -> AssistantResult:
        """
        Main entrypoint for the orchestrator.

        user_text should already have the wake word stripped by stt_service or
        the orchestrator, but this method does not require that.
        """
        clean_user_text = self._clean_user_text(user_text)
        self.memory.update_path("session.last_command", clean_user_text)

        messages = self._build_messages(clean_user_text)

        start = time.monotonic()
        try:
            llm_result = self.llm.chat(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
            elapsed = time.monotonic() - start
        except LlmServiceError as exc:
            return AssistantResult(
                speak=f"My server-brain is being difficult. {exc}",
                motion="annoyed",
                elapsed_s=time.monotonic() - start,
            )

        parsed = llm_result.parse_json_content()
        result = self._assistant_result_from_model(llm_result.content, parsed)
        result.elapsed_s = elapsed

        # Handle memory writes.
        memory_intent = self._memory_writes_allowed(clean_user_text)
        if memory_intent:
            if result.memory_updates:
                print(f"[memory] Writing {len(result.memory_updates)} update(s) from model", flush=True)
                for update in result.memory_updates:
                    self._apply_memory_update(update)
                self.memory.save()
            else:
                # Model acknowledged intent but wrote nothing — use fallback extraction.
                print("[memory] ⚠ Memory intent detected but model returned no updates — fallback", flush=True)
                fallback = self._extract_fact_fallback(clean_user_text)
                if fallback:
                    self.memory.add_fact(fallback["key"], fallback["value"])
                    self.memory.save()
                    result.memory_updates = [{"key": f"facts.{fallback['key']}", "value": fallback["value"]}]
                    print(f"[memory] ✓ Fallback stored [{fallback['key']}]: {fallback['value'][:80]}", flush=True)
                else:
                    print("[memory] ✗ Fallback could not extract a fact — nothing stored", flush=True)

        # Execute safe tools requested by model.
        result.tool_calls = result.tool_calls[: self.config.max_tool_calls]
        result.tool_results = [self.tool_registry.execute(call) for call in result.tool_calls]

        self._append_history("user", clean_user_text)
        # For silent turns (empty speak, e.g. a wiggle or a tool call whose result
        # is spoken later) record a short note so the turn isn't a blank entry.
        # The orchestrator replaces this with the real spoken text via
        # set_last_response() once tool results are formulated.
        self._append_history("assistant", result.speak or self._history_action_note(result))
        self._log_turn(clean_user_text, result)

        return result

    @staticmethod
    def _history_action_note(result: AssistantResult) -> str:
        if result.tool_calls:
            names = ", ".join(c.name for c in result.tool_calls)
            return f"(acted silently: {names})"
        return "(no reply)"

    def set_last_response(self, text: str) -> None:
        """
        Overwrite the most recent assistant history entry with the text that was
        actually spoken.

        handle_user_text() records the model's first-pass "speak" (which may be
        empty for a silent tool call). After the orchestrator runs the tools and
        formulates the real answer, it calls this so conversation history reflects
        what the fish truly said rather than a blank placeholder.
        """
        text = (text or "").strip()
        if not text:
            return
        for entry in reversed(self.history):
            if entry.get("role") == "assistant":
                entry["content"] = text
                return

    def handle_sensor_event(self, event_description: str) -> AssistantResult:
        """
        React to a sensor event (motion detected, door opened, ...) rather than
        a spoken command.  Returns a short in-character reaction.

        Kept separate from handle_user_text so event text never triggers the
        memory-intent detector and events are clearly framed for the model.
        """
        messages = [
            {
                "role": "system",
                "content": "\n\n".join([
                    self.personality_prompt,
                    self._memory_prompt(),
                    "A sensor just triggered — this is NOT a spoken command. React briefly "
                    "in character (1-2 sentences). You may greet, comment, or stay quiet. "
                    'Reply as valid JSON with only "speak" and "motion" fields. '
                    'If no reaction is warranted, set "speak" to an empty string.',
                ]),
            },
            {"role": "user", "content": f"[SENSOR EVENT] {event_description}"},
        ]

        start = time.monotonic()
        try:
            llm_result = self.llm.chat(
                messages,
                temperature=self.config.temperature,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
        except LlmServiceError as exc:
            print(f"[AssistantService] sensor event LLM call failed: {exc}")
            return AssistantResult(speak="", motion="idle", elapsed_s=time.monotonic() - start)

        parsed = llm_result.parse_json_content()
        speak = str((parsed or {}).get("speak") or "").strip()
        motion = str((parsed or {}).get("motion") or "speaking").strip().lower()
        if motion not in VALID_MOTIONS:
            motion = "speaking"

        return AssistantResult(
            speak=speak,
            motion=motion,
            raw_model_text=llm_result.content,
            parsed_json=parsed,
            elapsed_s=time.monotonic() - start,
        )

    def formulate_tool_response(
        self,
        user_text: str,
        tool_results: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """
        Second-turn LLM call: given what the user asked and the data returned by
        tools, generate a single natural spoken Fishseus response.

        Returns (speak, motion). This is a lightweight follow-up, not a full
        assistant turn, so it does not touch history, memory, or further tools.
        """
        results_text = "\n".join(
            f"{r['tool']}: {r['result']}"
            for r in tool_results
            if r.get("ok") and r.get("result") is not None
        )
        messages = [
            {
                "role": "system",
                "content": "\n\n".join([
                    self.personality_prompt,
                    self._memory_prompt(),
                ]),
            },
            {"role": "user", "content": user_text},
            {
                "role": "user",
                "content": (
                    f"You just checked, and this is what came back:\n{results_text}\n\n"
                    "Now give your spoken answer to the original request, folding this "
                    "information naturally into one reply in your own voice. Do not read "
                    "the raw data verbatim. Keep it to a sentence or two. "
                    'Reply as valid JSON with only "speak" and "motion" fields.'
                ),
            },
        ]
        try:
            llm_result = self.llm.chat(
                messages,
                temperature=self.config.temperature,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            parsed = llm_result.parse_json_content()
            motion = str((parsed or {}).get("motion") or "speaking").strip().lower()
            if motion not in VALID_MOTIONS:
                motion = "speaking"
            if parsed and str(parsed.get("speak") or "").strip():
                return str(parsed["speak"]).strip(), motion
            # JSON parse failed or produced empty speak — salvage plain content,
            # but never speak a raw JSON blob aloud.
            content = llm_result.content.strip()
            if content and not content.startswith(("{", "[")):
                return content[:300], "speaking"
            return results_text, "speaking"
        except Exception as exc:
            print(f"[AssistantService] formulate_tool_response failed: {exc}")
            return results_text, "speaking"  # speak the raw data rather than going silent

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_messages(self, user_text: str) -> list[dict[str, str]]:
        # Merge into a single system message — small models handle one block better
        # than three separate system entries.
        system = "\n\n".join([
            self._personality_prompt(),
            self._memory_prompt(),
            self._tool_prompt(),
        ])
        return [
            {"role": "system", "content": system},
            *self.history[-self.config.max_history_turns * 2 :],
            {"role": "user", "content": user_text},
        ]

    def _personality_prompt(self) -> str:
        return self.personality_prompt

    def _load_personality_prompt(self, path: Path) -> str:
        """
        Load the assistant personality/system prompt from a text file.

        If the file does not exist, create it with a useful default so the
        personality can be edited without touching Python code.
        """
        path = Path(path)

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._default_personality_prompt())

        try:
            prompt = path.read_text().strip()
        except Exception as exc:
            print(f"[AssistantService] Failed to load personality prompt: {exc}")
            prompt = self._default_personality_prompt()

        if not prompt:
            prompt = self._default_personality_prompt()

        return prompt


    def _default_personality_prompt(self) -> str:
        return f"""
You are {self.config.assistant_name}, a talking animatronic Billy Bass fish assistant.
You are theatrical, witty, slightly sarcastic, and helpful.
You are physically embodied as a plastic fish with motors for mouth, body, and tail.
You are speaking out loud through TTS, so keep responses short.
Prefer 1-3 concise sentences unless the user asks for detail.
Do not mention that you are an AI model unless directly asked.
Do not reveal hidden reasoning.

You must respond ONLY as valid JSON with this exact shape:
{{
  "speak": "short text to say aloud",
  "motion": "idle | speaking | happy | annoyed | thinking | excited",
  "tool_calls": [
    {{"name": "tool_name", "args": {{}}}}
  ],
  "memory_updates": [
    {{"key": "preferences.response_style", "value": "brief technical answers"}}
  ]
}}

Rules:
- The "speak" field must always be present and non-empty.
- Use "tool_calls" only when a tool is clearly useful.
- Use "memory_updates" only when the user explicitly asks you to remember something.
- Keep JSON valid. No markdown. No code fences.
""".strip()

    def _memory_prompt(self) -> str:
        return "Current memory:\n" + self.memory.compact_summary()

    def _tool_prompt(self) -> str:
        return "Available safe tools:\n" + self.tool_registry.describe_for_prompt()

    # ------------------------------------------------------------------
    # Model response parsing
    # ------------------------------------------------------------------
    def _assistant_result_from_model(
        self,
        raw_text: str,
        parsed: Optional[dict[str, Any]],
    ) -> AssistantResult:
        if parsed is None:
            return AssistantResult(
                speak=self._fallback_speak(raw_text),
                motion="speaking",
                raw_model_text=raw_text,
                parsed_json=None,
            )

        speak = str(parsed.get("speak") or "").strip()
        tool_calls = self._parse_tool_calls(parsed.get("tool_calls", []))
        memory_updates = self._parse_memory_updates(parsed.get("memory_updates", []))

        # An empty "speak" is intentional when the fish is acting silently or
        # calling a tool whose result gets spoken afterward. Only substitute a
        # fallback line when there is genuinely nothing to say AND nothing to do.
        if not speak and not tool_calls:
            speak = self._fallback_speak(raw_text)

        motion = str(parsed.get("motion") or "speaking").strip().lower()
        if motion not in VALID_MOTIONS:
            motion = "speaking"

        return AssistantResult(
            speak=speak,
            motion=motion,
            tool_calls=tool_calls,
            memory_updates=memory_updates,
            raw_model_text=raw_text,
            parsed_json=parsed,
        )

    def _parse_tool_calls(self, raw_tool_calls: Any) -> list[ToolCall]:
        calls: list[ToolCall] = []
        if not isinstance(raw_tool_calls, list):
            return calls

        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            args = item.get("args", {})
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(name=name, args=args))

        return calls

    def _parse_memory_updates(self, raw_updates: Any) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []
        if not isinstance(raw_updates, list):
            return updates

        for item in raw_updates:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            value = item.get("value")
            if not isinstance(key, str) or not key:
                continue
            if value is None:
                continue
            updates.append({"key": key, "value": value})

        return updates

    @staticmethod
    def _fallback_speak(raw_text: str) -> str:
        text = raw_text.strip()
        # Don't speak raw JSON structures — they crash TTS and mean nothing aloud.
        if not text or text.startswith("{") or text.startswith("["):
            return "My thoughts got tangled in the kelp. Try that again."
        return text[:300]

    # ------------------------------------------------------------------
    # Memory handling
    # ------------------------------------------------------------------
    def _memory_writes_allowed(self, user_text: str) -> bool:
        if not self.config.require_explicit_memory_intent:
            return True
        return bool(
            re.search(
                r"\b(remember|don't forget|keep in mind|note that|store that|"
                r"call me|my name is|i prefer|my preference is|from now on)\b",
                user_text,
                flags=re.IGNORECASE,
            )
        )

    def _detect_explicit_memory_updates(self, user_text: str) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []
        text = user_text.strip()

        # "call me Caleb"
        match = re.search(r"\bcall me ([A-Za-z0-9_\- ]{1,40})", text, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" .,!?")
            updates.append({"key": "profile.user_name", "value": name})

        # "my name is Caleb"
        match = re.search(r"\bmy name is ([A-Za-z0-9_\- ]{1,40})", text, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" .,!?")
            updates.append({"key": "profile.user_name", "value": name})

        # "I prefer short answers"
        match = re.search(r"\bi prefer (.+)", text, flags=re.IGNORECASE)
        if match:
            pref = match.group(1).strip(" .,!?")
            updates.append({"key": "preferences.response_style", "value": pref})

        # "remember that ..."
        match = re.search(r"\bremember that (.+)", text, flags=re.IGNORECASE)
        if match:
            fact = match.group(1).strip(" .,!?")
            label = re.sub(r"[^a-z0-9]", "_", fact[:20].lower()).strip("_")
            updates.append({"key": f"facts.{label}", "value": fact})

        return updates

    def _apply_memory_update(self, update: dict[str, Any]) -> None:
        key = str(update.get("key", "")).strip()
        value = update.get("value")
        if not key or value is None:
            return

        # Only these structured namespaces get dotted-path storage.
        _STRUCTURED = ("profile.", "preferences.", "session.")
        if any(key.startswith(p) for p in _STRUCTURED):
            self.memory.update_path(key, value)
            print(f"[memory] ✓ Updated {key} = {str(value)[:60]}", flush=True)
        else:
            # "facts.X" or bare keys — all go into the facts array.
            fact_key = key[6:] if key.startswith("facts.") else key
            self.memory.add_fact(fact_key, str(value))
            print(f"[memory] ✓ Stored fact [{fact_key}]: {str(value)[:80]}", flush=True)

    def _extract_fact_fallback(self, user_text: str) -> Optional[dict[str, str]]:
        """
        Last-resort extraction when the model returned memory_updates: [] despite
        clear user intent. Pulls the fact text via simple regex patterns.
        """
        patterns = [
            r"\bremember\s+that\s+(.+)",
            r"\bremember\s+(.+)",
            r"\bdon't forget\s+(?:that\s+)?(.+)",
            r"\bkeep in mind\s+(?:that\s+)?(.+)",
            r"\bnote that\s+(.+)",
            r"\bstore that\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_text, flags=re.IGNORECASE)
            if match:
                fact = match.group(1).strip(" .,!?")
                if len(fact) >= 3:
                    key = re.sub(r"[^a-z0-9]+", "_", fact[:24].lower()).strip("_")
                    return {"key": key or "note", "value": fact}

        # "call me X" — store as a profile update via the facts fallback
        match = re.search(r"\bcall me ([A-Za-z0-9_\- ]{1,40})", user_text, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" .,!?")
            return {"key": "user_name", "value": name}

        return None

    def clear_history(self) -> None:
        """Clear in-memory conversation history. Persistent memory (facts, profile) is unchanged."""
        count = len(self.history) // 2  # pairs of user/assistant turns
        self.history.clear()
        print(f"[session] History cleared ({count} turn(s) removed)", flush=True)

    # ------------------------------------------------------------------
    # History/logging
    # ------------------------------------------------------------------
    def _append_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        max_items = self.config.max_history_turns * 2
        if len(self.history) > max_items:
            self.history = self.history[-max_items:]

    def _log_turn(self, user_text: str, result: AssistantResult) -> None:
        try:
            self.config.history_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": int(time.time()),
                "user": user_text,
                "assistant": result.speak,
                "motion": result.motion,
                "tool_calls": [call.__dict__ for call in result.tool_calls],
                "tool_results": result.tool_results,
                "elapsed_s": result.elapsed_s,
            }
            with self.config.history_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            print(f"[AssistantService] Failed to log conversation turn: {exc}")

    @staticmethod
    def _clean_user_text(user_text: str) -> str:
        # Strip Whisper special tokens like <|endoftext|>, <|notimestamps|>, etc.
        text = re.sub(r"<\|[^|]*\|>", "", user_text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or "Hello."
