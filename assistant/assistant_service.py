"""
assistant_service.py

The assistant brain for the Fishseus/Billy Bass assistant project.

Responsibilities:
- Own the personality prompt and the editable prompt text (assistant/prompts.py).
- Own long-term memory and the recalled conversation (assistant/memory.py).
- Build messages for llm_service.py and parse structured model responses.
- Validate and execute safe local tool calls, then get the spoken answer.

Non-responsibilities:
- Does not record audio, run STT, or play TTS.
- Does not directly own GPIO unless tools are wired in by the orchestrator.
- Does not know what model/server is being used beyond the LlmService interface.

Expected flow:
    assistant = AssistantService(llm=llm, tool_registry=tools)
    result = assistant.handle_user_text("what's the weather?", say=speak_fn)
    print(result.spoken_text)

`say(text, motion, final)` is called for each line as it should be spoken: a
short line while a tool runs (final=False), then the answer (final=True). The
web UI passes no `say` and reads `result.spoken_text` instead.

What the model sees each turn:
    system:  personality + response rules + available tools (with examples)
             + long-term memory
    ...the recalled conversation (last `recall_hours`), replayed in the same
       JSON reply format the model must produce, including tool calls/results
    system:  the current date and time
    user:    the new message
The stable parts come first so providers can cache the prompt prefix.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import re
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Optional

from assistant.memory import ConversationStore, MemoryStore, is_placeholder_name, normalize_key
from assistant.prompts import PromptTexts
from llm.llm_service import LlmService, LlmServiceError
from services import Service, ServiceConfig, ServiceError, ROOT_DIR


class AssistantServiceError(ServiceError):
    pass


# Motions the model may request. Anything outside this set falls back to "speaking".
VALID_MOTIONS = {"idle", "speaking", "happy", "annoyed", "thinking", "excited"}

# A pause at least this long in the recalled conversation gets a time marker.
_GAP_MARKER_S = 15 * 60

_SENSOR_PREFIX = "[Sensor event, not speech]"
_TOOL_RESULTS_HEADER = "[Tool results]"


# ----------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class AssistantConfig(ServiceConfig):
    module_name: str = "assistant"

    assistant_name: str = "Fishseus"
    # Seeds long-term memory until the human says their name.
    user_name: str = ""

    personality_path: Path = ROOT_DIR / "config" / "personality_prompt.txt"
    prompt_texts_path: Path = ROOT_DIR / "config" / "prompt_texts.json"
    memory_path: Path = ROOT_DIR / "data" / "assistant_memory.json"
    history_path: Path = ROOT_DIR / "data" / "conversation_log.jsonl"

    # Every turn is saved to history_path. Turns from the last `recall_hours`
    # (0 = no time limit) are replayed into each prompt, newest
    # `max_recall_turns` at most (0 = no cap).
    recall_hours: float = 12.0
    max_recall_turns: int = 150

    max_tool_calls: int = 3

    temperature: float = 0.7
    max_tokens: int = 350

    # If true, new memories are only saved when the human asks ("remember",
    # "save", "call me", ...). Corrections to an existing fact are always allowed.
    require_explicit_memory_intent: bool = True

    def validate(self) -> bool:
        for name in ("recall_hours", "max_recall_turns", "max_tool_calls"):
            if getattr(self, name) < 0:
                raise AssistantServiceError(f"{name} must be >= 0: {getattr(self, name)}")
        return True

    @classmethod
    def from_section(cls, section: dict, *, config_dir: Path, **overrides: Any) -> "AssistantConfig":
        """
        Build from the "assistant" section of fish_config.json. Path settings
        are relative to the config directory; unknown keys are ignored so old
        configs keep working.
        """
        known = {f.name for f in fields(cls)} - {"module_name"}
        values = {k: v for k, v in (section or {}).items() if k in known}
        for key in ("personality_path", "prompt_texts_path", "memory_path", "history_path"):
            if key in values:
                values[key] = (Path(config_dir) / values[key]).resolve()
        values.update(overrides)
        return cls(**values)


# Settings the web UI may change on a running assistant.
LIVE_SETTINGS: dict[str, type] = {
    "recall_hours": float,
    "max_recall_turns": int,
    "max_tool_calls": int,
    "temperature": float,
    "max_tokens": int,
    "require_explicit_memory_intent": bool,
}


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantResult:
    speak: str                      # the model's first reply ("" = acted silently)
    motion: str = "speaking"
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    memory_updates: list[dict[str, Any]] = field(default_factory=list)  # as actually stored
    answer: str = ""                # spoken after tool results came back
    answer_motion: str = ""
    spoken_text: str = ""           # everything said aloud this turn
    raw_model_text: str = ""
    parsed_json: Optional[dict[str, Any]] = None
    elapsed_s: float = 0.0


def _always_available() -> bool:
    return True


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
    hint: str = ""                  # when/how to use it, shown to the model with the description
    # Few-shot examples shown to the model while the tool is usable:
    # [{"user": "what is 47 times 83?", "args": {"expression": "47 * 83"},
    #   "speak": "" (optional), "motion": "thinking" (optional)}]
    examples: list[dict[str, Any]] = field(default_factory=list)
    # False while the service behind the tool isn't running (hidden + not executable).
    available: Callable[[], bool] = _always_available


# Tool fields the web UI can override; overrides persist across restarts.
_OVERRIDABLE = ("enabled", "description", "hint", "examples", "synthesize_result")


def normalize_examples(raw: Any) -> list[dict[str, Any]]:
    """Validate tool examples from the web UI / overrides file. Raises ValueError."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("examples must be a list")
    examples = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"example {i} must be an object")
        user = str(item.get("user") or "").strip()
        if not user:
            raise ValueError(f"example {i} needs what the user says")
        args = item.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as exc:
                raise ValueError(f"example {i}: args must be JSON ({exc.msg})") from exc
        if not isinstance(args, dict):
            raise ValueError(f"example {i}: args must be a JSON object")
        example: dict[str, Any] = {"user": user, "args": args}
        speak = str(item.get("speak") or "").strip()
        if speak:
            example["speak"] = speak
        motion = str(item.get("motion") or "").strip().lower()
        if motion:
            if motion not in VALID_MOTIONS:
                raise ValueError(f"example {i}: motion must be one of {sorted(VALID_MOTIONS)}")
            example["motion"] = motion
        examples.append(example)
    return examples


def reply_json(speak: str, motion: str, tool_calls: list[Any], memory_updates: list[Any]) -> str:
    """The exact reply shape the model must produce (used for examples and replay)."""
    calls = []
    for call in tool_calls or []:
        if isinstance(call, ToolCall):
            calls.append({"name": call.name, "args": call.args})
        elif isinstance(call, dict) and call.get("name"):
            calls.append({"name": call["name"], "args": call.get("args") or {}})
    updates = [
        {"key": u.get("key"), "value": u.get("value")}
        for u in memory_updates or [] if isinstance(u, dict) and u.get("key")
    ]
    return json.dumps(
        {"speak": speak or "", "motion": motion or "speaking", "tool_calls": calls, "memory_updates": updates},
        ensure_ascii=False,
        default=str,
    )


# ----------------------------------------------------------------------
# Tool registry
# ----------------------------------------------------------------------

class ToolRegistry:
    """
    Stores safe callable tools.

    The model can request tool calls, but this registry decides what actually
    executes. The LLM never directly controls hardware.

    With an overrides_path, web UI edits (enable/disable, description, hint,
    examples, synthesize flag) are saved there as differences from the code
    defaults and re-applied at start-up.
    """

    def __init__(self, overrides_path: Optional[Path] = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._defaults: dict[str, dict[str, Any]] = {}
        self._overrides_path = Path(overrides_path) if overrides_path else None
        self._overrides: dict[str, dict[str, Any]] = self._load_overrides()
        self._lock = threading.RLock()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._defaults[tool.name] = {f: copy.deepcopy(getattr(tool, f)) for f in _OVERRIDABLE}
        for key, value in self._overrides.get(tool.name, {}).items():
            if key not in _OVERRIDABLE:
                continue
            try:
                if key == "examples":
                    value = normalize_examples(value)
                setattr(tool, key, value)
            except ValueError as exc:
                print(f"[tools] Ignoring bad override {tool.name}.{key}: {exc}", flush=True)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def usable(self) -> list[Tool]:
        """Tools the model may call right now."""
        result = []
        for tool in self._tools.values():
            if tool.risk != "safe" or not tool.enabled:
                continue
            try:
                if not tool.available():
                    continue
            except Exception:
                continue
            result.append(tool)
        return result

    def describe_for_prompt(self) -> str:
        tools = self.usable()
        if not tools:
            return "AVAILABLE TOOLS\nNone right now. Don't call any tools."
        lines = ["AVAILABLE TOOLS"]
        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description}")
            if tool.hint:
                lines.append(f"  Use it: {tool.hint}")
            for example in tool.examples:
                motion = example.get("motion") or ("thinking" if tool.returns_data else "speaking")
                call = ToolCall(tool.name, example.get("args") or {})
                lines.append(f'  User: "{example["user"]}"')
                lines.append("  " + reply_json(example.get("speak", ""), motion, [call], []))
        return "\n".join(lines)

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
            available = tool.available()
        except Exception:
            available = False
        if not available:
            return {"tool": call.name, "ok": False, "error": "The hardware or service behind this tool is offline"}

        # Drop arguments the function doesn't take; models occasionally invent them.
        args = dict(call.args or {})
        try:
            params = inspect.signature(tool.function).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                args = {k: v for k, v in args.items() if k in params}
        except (TypeError, ValueError):
            pass

        try:
            result = tool.function(**args)
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

    def update(self, name: str, **changes: Any) -> None:
        """Change overridable fields and persist them. Raises KeyError/ValueError."""
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        for key, value in changes.items():
            if key not in _OVERRIDABLE:
                raise ValueError(f"'{key}' can't be edited")
            if key == "examples":
                value = normalize_examples(value)
            elif key in ("enabled", "synthesize_result"):
                value = bool(value)
            else:
                value = str(value or "").strip()
                if key == "description" and not value:
                    raise ValueError("description can't be empty")
            setattr(tool, key, value)
        self._store_override(name)

    def reset(self, name: str) -> None:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        for key, value in self._defaults[name].items():
            setattr(tool, key, copy.deepcopy(value))
        self._store_override(name)

    def enable(self, name: str) -> None:
        if name in self._tools:
            self.update(name, enabled=True)

    def disable(self, name: str) -> None:
        if name in self._tools:
            self.update(name, enabled=False)

    def update_description(self, name: str, description: str) -> None:
        if name in self._tools:
            self.update(name, description=description)

    def set_synthesize_result(self, name: str, value: bool) -> None:
        if name in self._tools:
            self.update(name, synthesize_result=value)

    def list_all(self) -> list[dict]:
        result = []
        for t in self._tools.values():
            try:
                available = bool(t.available())
            except Exception:
                available = False
            result.append({
                "name": t.name,
                "description": t.description,
                "hint": t.hint,
                "examples": copy.deepcopy(t.examples),
                "risk": t.risk,
                "enabled": t.enabled,
                "available": available,
                "returns_data": t.returns_data,
                "synthesize_result": t.synthesize_result,
                "customized": bool(self._overrides.get(t.name)),
                "defaults": copy.deepcopy(self._defaults.get(t.name, {})),
            })
        return result

    # ------------------------------------------------------------------
    # Overrides file
    # ------------------------------------------------------------------

    def _load_overrides(self) -> dict[str, dict[str, Any]]:
        path = self._overrides_path
        if path is None or not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[tools] Could not read {path.name}, ignoring tool overrides: {exc}", flush=True)
            return {}
        return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}

    def _store_override(self, name: str) -> None:
        with self._lock:
            tool = self._tools[name]
            diff = {
                key: copy.deepcopy(getattr(tool, key))
                for key, default in self._defaults[name].items()
                if getattr(tool, key) != default
            }
            if diff:
                self._overrides[name] = diff
            else:
                self._overrides.pop(name, None)
            path = self._overrides_path
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._overrides, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            tmp.replace(path)


# ----------------------------------------------------------------------
# Memory intent
# ----------------------------------------------------------------------

# Phrases that mean "keep this". Deliberately broad; the model still decides
# what (if anything) to store.
_MEMORY_CUE = re.compile(
    r"\b(remember|memori[sz]e|memory|don'?t forget|do not forget|keep in mind|"
    r"make a note|take a note|note (?:that|this|down)|jot (?:that |this |it )?down|"
    r"write (?:that |this |it )?down|(?:save|store|log|add|put) (?:that|this|it)\b|"
    r"add (?:that |this |it )?to your|call me|my name is|my name's|i prefer|"
    r"my preference|from now on)",
    flags=re.IGNORECASE,
)

_QUESTION_START = re.compile(
    r"^(?:(?:hey|so|okay|ok|well|and|fish|fishseus)[,\s]+)*"
    r"(what|what's|who|who's|whose|where|when|why|how|which|do|does|did|can|could|"
    r"would|will|is|are|was|were|have|has)\b",
    flags=re.IGNORECASE,
)

# Last-resort extraction when the model ignored an explicit request.
# (pattern, allowed even when the utterance reads as a question)
_FALLBACK_PATTERNS: list[tuple[re.Pattern[str], bool]] = [
    (re.compile(r"\bremember\s+that\s+(.+)", re.I), True),
    (re.compile(r"\b(?:don'?t|do not)\s+forget\s+(?:that\s+)?(.+)", re.I), True),
    (re.compile(r"\bkeep in mind\s+(?:that\s+)?(.+)", re.I), True),
    (re.compile(r"\b(?:make a |take a )?note\s+(?:that|down)\s+(.+)", re.I), True),
    (re.compile(r"\b(?:add|save|store|write|put)\b.{0,30}?\b(?:memory|facts|notes|down)\b\s+that\s+(.+)", re.I), True),
    (re.compile(r"\bremember[,:]?\s+(.+)", re.I), False),
]

# Requests that stay requests even when phrased as a question.
_REQUEST_IN_QUESTION = re.compile(
    r"\b(?:remember|note|save|store|keep in mind|write down)\s+(?:that|this)\b|"
    r"\badd (?:that |this |it )?to your\b|\bcall me\b|\bmy name(?: is|'s)\b",
    flags=re.IGNORECASE,
)

# The human's name only changes when they actually give one.
_NAME_CUE = re.compile(r"\b(?:call me|my name|i am|i'm|name is|called)\b", re.IGNORECASE)

_NAME_PATTERN = re.compile(r"\b(?:call me|my name is|my name's)\s+([A-Za-z][A-Za-z\-' ]{0,39})", re.I)

# "Remember this", "remember everything": nothing concrete to store.
_VAGUE_FACT = re.compile(
    r"^(?:this|that|it|everything|all of (?:this|that|it)|what (?:we|i|you) (?:said|talked about|discussed)|"
    r"(?:this|that|our|the) (?:conversation|chat|talk|discussion))\b[\s\w]{0,12}$",
    re.IGNORECASE,
)

_KEY_STOPWORDS = {"the", "a", "an", "is", "are", "was", "that", "my", "his", "her", "their", "of", "at", "in", "on"}

_NAME_KEYS = {"user_name", "my_name", "name", "username", "human_name"}


# ----------------------------------------------------------------------
# Assistant service
# ----------------------------------------------------------------------

SayFn = Callable[[str, str, bool], None]  # (text, motion, is_final_line)


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
        self.conversation = ConversationStore(
            config.history_path, config.recall_hours, config.max_recall_turns)
        self.conversation.load()
        self.texts = PromptTexts(config.prompt_texts_path)

        self._personality_mtime: Optional[float] = None
        self.personality_prompt = self._load_personality_prompt(config.personality_path)

        self.tool_registry = tool_registry or ToolRegistry()
        # One conversational turn at a time: the voice loop and the web chat
        # share one conversation, so their turns must not interleave.
        self._turn_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        # __init__ already loads everything; re-run here (idempotent) so the
        # orchestrator can drive every service the same way.
        self.config.validate()
        self.memory.load()
        self.conversation.load()
        self.texts.load()
        self.personality_prompt = self._load_personality_prompt(self.config.personality_path)
        recalled = len(self.conversation.recent())
        print(f"[assistant] Long-term memory: {len(self.memory.facts())} fact(s). "
              f"Recalling {recalled} turn(s) from the last {self.config.recall_hours:g}h.", flush=True)

    def shutdown(self) -> None:
        try:
            self.memory.save()
        except Exception as exc:
            print(f"[AssistantService] memory save failed: {exc}")

    def reset(self) -> bool:
        self.memory.load()
        self.conversation.load()
        self.texts.load()
        return True

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "service": "ok",
            "recalled_turns": len(self.conversation.recent()),
            "facts": len(self.memory.facts()),
            "tools": len(self.tool_registry.list_all()),
        }

    def apply_settings(self, values: dict[str, Any]) -> None:
        """Apply web UI settings to the running assistant. Raises on bad values."""
        changes: dict[str, Any] = {}
        for key, cast in LIVE_SETTINGS.items():
            if key not in values or values[key] is None or values[key] == "":
                continue
            value = values[key]
            if cast is bool and isinstance(value, str):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            changes[key] = cast(value)
        if changes:
            new_config = dataclasses.replace(self.config, **changes)
            new_config.validate()
            self.config = new_config
            self.conversation.configure(new_config.recall_hours, new_config.max_recall_turns)

        profile_changed = False
        if values.get("assistant_name"):
            self.memory.set_profile("assistant_name", values["assistant_name"])
            profile_changed = True
        if "user_name" in values and not is_placeholder_name(values["user_name"]):
            self.memory.set_profile("user_name", values["user_name"])
            profile_changed = True
        if profile_changed:
            self.memory.save()

    # ------------------------------------------------------------------
    # Conversation entry points
    # ------------------------------------------------------------------
    def handle_user_text(
        self,
        user_text: str,
        *,
        source: str = "voice",
        say: Optional[SayFn] = None,
    ) -> AssistantResult:
        """
        Run one full turn: ask the model, store any memories, run tools, get the
        spoken answer from tool results, and record the turn.

        user_text should already have the wake word stripped by stt_service or
        the orchestrator, but this method does not require that.
        """
        clean_user_text = self._clean_user_text(user_text)

        with self._turn_lock:
            history = self.conversation.recent()
            messages = self._build_messages(history, [{"role": "user", "content": clean_user_text}])

            start = time.monotonic()
            try:
                llm_result = self._chat(messages)
            except LlmServiceError as exc:
                print(f"[assistant] LLM request failed: {exc}", flush=True)
                result = AssistantResult(
                    speak=self.texts.get("fallback_offline"),
                    motion="annoyed",
                    elapsed_s=time.monotonic() - start,
                )
                result.spoken_text = result.speak
                self._say(say, result.speak, result.motion, final=True)
                return result  # a failed request isn't part of the conversation
            elapsed = time.monotonic() - start

            parsed = llm_result.parse_json_content()
            result = self._assistant_result_from_model(llm_result.content, parsed)
            result.elapsed_s = elapsed

            proposed_memory = bool(result.memory_updates)
            result.memory_updates = self._apply_memory_updates(
                clean_user_text, result.memory_updates, used_tools=bool(result.tool_calls))
            if proposed_memory and not result.speak and not result.tool_calls:
                # A silent save leaves the human wondering; confirm only what was really stored.
                result.speak = self.texts.get("memory_saved" if result.memory_updates else "fallback_error")

            result.tool_calls = result.tool_calls[: self.config.max_tool_calls]
            result.tool_results = [self.tool_registry.execute(call) for call in result.tool_calls]

            self._finish_turn(history, clean_user_text, result, say)
            self._record_turn(source, clean_user_text, result)
            return result

    def handle_sensor_event(self, event_description: str) -> AssistantResult:
        """
        React to a sensor event (motion detected, door opened, ...) rather than
        a spoken command. Returns a short in-character reaction for the caller
        to speak; a reaction that was spoken is recorded in the conversation.

        Kept separate from handle_user_text so event text never triggers the
        memory-intent detector or tools.
        """
        with self._turn_lock:
            history = self.conversation.recent()
            prompt = f"{_SENSOR_PREFIX} {event_description}\n\n{self.texts.get('sensor_event')}"
            messages = self._build_messages(history, [{"role": "user", "content": prompt}])

            start = time.monotonic()
            try:
                llm_result = self._chat(messages, max_tokens=150)
            except LlmServiceError as exc:
                print(f"[AssistantService] sensor event LLM call failed: {exc}")
                return AssistantResult(speak="", motion="idle", elapsed_s=time.monotonic() - start)

            parsed = llm_result.parse_json_content() or {}
            speak = self._unwrap_speak(str(parsed.get("speak") or ""))
            result = AssistantResult(
                speak=speak,
                motion=self._valid_motion(parsed.get("motion")),
                spoken_text=speak,
                raw_model_text=llm_result.content,
                parsed_json=parsed or None,
                elapsed_s=time.monotonic() - start,
            )
            if speak:
                self._record_turn("sensor", event_description, result)
            return result

    def clear_history(self) -> None:
        """Start fresh: stop recalling the conversation so far. Long-term memory is unchanged."""
        dropped = self.conversation.clear()
        print(f"[session] Starting fresh ({dropped} turn(s) no longer recalled)", flush=True)

    def remember_conversation(self, focus: str = "") -> str:
        """
        Boil the recalled conversation down to a dated summary plus lasting facts
        and save them to long-term memory, so they outlive the recall window.
        Returns a report for the fish to confirm in its own voice; raises
        AssistantServiceError when it can't.
        """
        turns = self.conversation.recent()
        if not turns:
            return "There's no recent conversation to remember yet."

        known = "\n".join(f"- [{f['key']}] {f['value']}" for f in self.memory.facts()) or "(nothing yet)"
        request = f"CONVERSATION\n{self._transcript(turns)}\n\nALREADY IN LONG-TERM MEMORY\n{known}"
        if focus.strip():
            request += f"\n\nThe human especially wants you to keep: {focus.strip()}"
        messages = [
            {"role": "system", "content": self.texts.get("conversation_summary")},
            {"role": "user", "content": request},
        ]
        try:
            llm_result = self._chat(messages, max_tokens=max(self.config.max_tokens, 800))
        except LlmServiceError as exc:
            raise AssistantServiceError(f"couldn't summarize the conversation: {exc}") from exc
        parsed = llm_result.parse_json_content()
        if not parsed:
            raise AssistantServiceError("the summary came back garbled")

        saved: list[str] = []
        summary = " ".join(str(parsed.get("summary") or "").split())
        if summary:
            # One summary per session: remembering again later updates it in place.
            started = float(turns[0].get("timestamp") or time.time())
            key = "conversation_" + time.strftime("%Y_%m_%d_%H%M", time.localtime(started))
            self.memory.remember(key, f"Conversation on {self._format_time(started, with_year=True)}: {summary}")

        facts = parsed.get("facts") if isinstance(parsed.get("facts"), list) else []
        for fact in facts[:15]:
            if not isinstance(fact, dict):
                continue
            status, key = self.memory.remember(str(fact.get("key") or ""), fact.get("value"))
            if status in ("added", "updated"):
                saved.append(f"{fact.get('value')} ({status})")
        self.memory.save()
        print(f"[memory] Remembered conversation of {len(turns)} turn(s): {len(saved)} fact(s) + summary",
              flush=True)

        report = f"Saved a summary of the conversation to long-term memory: {summary or '(no summary)'}"
        if saved:
            report += " Facts kept: " + " ".join(saved)
        else:
            report += " No new individual facts needed saving."
        return report

    def _transcript(self, turns: list[dict[str, Any]]) -> str:
        """Plain-text transcript of recorded turns (for summarizing)."""
        human = self.memory.user_name or "Human"
        fish = self.config.assistant_name
        lines = []
        for turn in turns:
            stamp = self._format_time(float(turn.get("timestamp") or 0))
            speaker = "Sensor" if turn.get("source") == "sensor" else human
            lines.append(f"[{stamp}] {speaker}: {turn.get('user', '')}")
            for r in turn.get("tool_results") or []:
                if r.get("ok") and r.get("returns_data"):
                    lines.append(f"    ({r.get('tool')} returned: {r.get('result')})")
            for u in turn.get("memory_updates") or []:
                lines.append(f"    (saved to memory: {u.get('value')})")
            lines.append(f"{fish}: {turn.get('assistant') or '(acted silently)'}")
        return "\n".join(lines)

    def context_preview(self, user_text: str = "(your next message goes here)") -> list[dict[str, str]]:
        """The exact message list the model would get for the next message."""
        return self._build_messages(self.conversation.recent(), [{"role": "user", "content": user_text}])

    # ------------------------------------------------------------------
    # Turn mechanics
    # ------------------------------------------------------------------
    def _chat(self, messages: list[dict[str, str]], max_tokens: Optional[int] = None):
        return self.llm.chat(
            messages,
            temperature=self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            response_format={"type": "json_object"},
        )

    @staticmethod
    def _say(say: Optional[SayFn], text: str, motion: str, *, final: bool) -> None:
        if say is not None and text:
            say(text, motion, final)

    @staticmethod
    def _needs_followup(tool_result: dict[str, Any]) -> bool:
        """Data came back to answer with, or the tool failed and the fish should say so."""
        if not tool_result.get("ok"):
            return True
        return bool(tool_result.get("returns_data")) and tool_result.get("result") is not None

    def _finish_turn(
        self,
        history: list[dict[str, Any]],
        user_text: str,
        result: AssistantResult,
        say: Optional[SayFn],
    ) -> None:
        """Speak the reply, fetching a follow-up answer when tools returned something."""
        followup = [r for r in result.tool_results if self._needs_followup(r)]
        spoken: list[str] = []

        if followup:
            # A line said while "checking" goes first; the answer follows it.
            if result.speak:
                self._say(say, result.speak, result.motion, final=False)
                spoken.append(result.speak)

            ok_data = [r for r in followup if r.get("ok")]
            # Short raw values (time, dice, song names) can be spoken as-is only
            # when the fish already said its own line; everything else gets
            # folded into a real sentence by the model.
            speak_raw = (
                result.speak
                and len(ok_data) == len(followup)
                and not any(r.get("synthesize_result") for r in ok_data)
            )
            if speak_raw:
                answer = " ".join(str(r["result"]) for r in ok_data).strip()
                answer_motion = "speaking"
            else:
                answer, answer_motion = self._formulate(history, user_text, result, followup)
            if not answer:
                answer = self.texts.get("fallback_error")
            result.answer, result.answer_motion = answer, answer_motion
            self._say(say, answer, answer_motion, final=True)
            spoken.append(answer)

        elif result.speak:
            self._say(say, result.speak, result.motion, final=True)
            spoken.append(result.speak)

        result.spoken_text = " ".join(spoken)

    def _formulate(
        self,
        history: list[dict[str, Any]],
        user_text: str,
        result: AssistantResult,
        tool_results: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Second LLM call: turn tool results into one spoken answer, in context."""
        messages = self._build_messages(history, [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply_json(
                result.speak, result.motion, result.tool_calls, result.memory_updates)},
            {"role": "user", "content": self._tool_results_text(tool_results)
                + "\n\n" + self.texts.get("tool_followup")},
        ])
        plain = self._plain_results(tool_results)
        try:
            llm_result = self._chat(messages)
        except LlmServiceError as exc:
            print(f"[assistant] Follow-up after tools failed: {exc}", flush=True)
            return plain, "speaking"  # speak the raw data rather than going silent

        parsed = llm_result.parse_json_content()
        motion = self._valid_motion((parsed or {}).get("motion"))
        speak = self._unwrap_speak(str((parsed or {}).get("speak") or ""))
        if speak:
            return speak, motion
        # JSON parse failed or produced empty speak — salvage plain content,
        # but never speak a raw JSON blob aloud.
        content = llm_result.content.strip()
        if content and not content.startswith(("{", "[")):
            return content[:400], "speaking"
        return plain, "speaking"

    @staticmethod
    def _plain_results(tool_results: list[dict[str, Any]]) -> str:
        return " ".join(str(r["result"]) for r in tool_results if r.get("ok") and r.get("result") is not None)

    @staticmethod
    def _tool_results_text(tool_results: list[dict[str, Any]]) -> str:
        lines = [_TOOL_RESULTS_HEADER]
        for r in tool_results:
            if r.get("ok"):
                lines.append(f"{r.get('tool')}: {r.get('result')}")
            else:
                lines.append(f"{r.get('tool')}: FAILED ({r.get('error', 'unknown error')})")
        return "\n".join(lines)

    def _record_turn(self, source: str, user_text: str, result: AssistantResult) -> None:
        def compact(r: dict[str, Any]) -> dict[str, Any]:
            out = {"tool": r.get("tool"), "ok": bool(r.get("ok"))}
            if r.get("ok"):
                out["result"] = r.get("result")
                out["returns_data"] = bool(r.get("returns_data"))
            else:
                out["error"] = r.get("error")
            return out

        self.conversation.append({
            "source": source,
            "user": user_text,
            "speak": result.speak,
            "motion": result.motion,
            "tool_calls": [{"name": c.name, "args": c.args} for c in result.tool_calls],
            "tool_results": [compact(r) for r in result.tool_results],
            "memory_updates": [{"key": u["key"], "value": u["value"]} for u in result.memory_updates],
            "answer": result.answer,
            "answer_motion": result.answer_motion,
            "assistant": result.spoken_text,
            "elapsed_s": round(result.elapsed_s, 3),
        })

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_messages(
        self,
        history: list[dict[str, Any]],
        tail: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        # One system block up front — stable across turns, so it caches well —
        # then the recalled conversation, then the clock, then this turn.
        return [
            {"role": "system", "content": self._system_prompt()},
            *self._history_messages(history),
            {"role": "system", "content": f"[Now: {self._format_time(time.time(), with_year=True)}]"},
            *tail,
        ]

    def _system_prompt(self) -> str:
        parts = [
            self._personality_prompt(),
            self.texts.get("rules"),
            self.tool_registry.describe_for_prompt(),
            self.memory.prompt_block(),
        ]
        return "\n\n".join(p.strip() for p in parts if p and p.strip())

    def _history_messages(self, turns: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        previous: Optional[float] = None
        for turn in turns:
            ts = float(turn.get("timestamp") or 0)
            if previous is None or ts - previous >= _GAP_MARKER_S:
                messages.append({"role": "system", "content": f"[{self._format_time(ts)}]"})
            messages += self._turn_messages(turn)
            previous = ts
        return messages

    def _turn_messages(self, turn: dict[str, Any]) -> list[dict[str, str]]:
        """Replay a recorded turn in the same format the model must produce."""
        user = str(turn.get("user") or "")
        if turn.get("source") == "sensor":
            user = f"{_SENSOR_PREFIX} {user}"
        messages = [{"role": "user", "content": user}]

        if "speak" not in turn:
            # Pre-v2 log line: only the spoken text survives.
            messages.append({"role": "assistant", "content": reply_json(
                str(turn.get("assistant") or ""), turn.get("motion") or "speaking",
                turn.get("tool_calls") or [], [])})
            return messages

        messages.append({"role": "assistant", "content": reply_json(
            str(turn.get("speak") or ""), turn.get("motion") or "speaking",
            turn.get("tool_calls") or [], turn.get("memory_updates") or [])})
        if turn.get("answer"):
            followup = [r for r in turn.get("tool_results") or [] if self._needs_followup(r)]
            messages.append({"role": "user", "content": self._tool_results_text(followup)})
            messages.append({"role": "assistant", "content": reply_json(
                str(turn["answer"]), turn.get("answer_motion") or "speaking", [], [])})
        return messages

    @staticmethod
    def _format_time(ts: float, *, with_year: bool = False) -> str:
        t = time.localtime(ts)
        hour = t.tm_hour % 12 or 12
        date = f"{time.strftime('%A, %B', t)} {t.tm_mday}"
        if with_year:
            date += f", {t.tm_year}"
        return f"{date}, {hour}:{t.tm_min:02d} {'AM' if t.tm_hour < 12 else 'PM'}"

    def _personality_prompt(self) -> str:
        """The personality file, re-read when it changes (web edits apply immediately)."""
        path = Path(self.config.personality_path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return self.personality_prompt
        if mtime != self._personality_mtime:
            self.personality_prompt = self._load_personality_prompt(path)
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
            path.write_text(self._default_personality_prompt(), encoding="utf-8")

        try:
            prompt = path.read_text(encoding="utf-8").strip()
            self._personality_mtime = path.stat().st_mtime
        except Exception as exc:
            print(f"[AssistantService] Failed to load personality prompt: {exc}")
            prompt = self._default_personality_prompt()

        if not prompt:
            prompt = self._default_personality_prompt()

        return prompt

    def _default_personality_prompt(self) -> str:
        return f"""
You are {self.config.assistant_name}, a talking animatronic Billy Bass fish assistant.
You are theatrical, witty, slightly sarcastic, and genuinely helpful.
You are physically embodied as a plastic fish with motors for mouth, body, and tail.
You are speaking out loud through TTS, so keep responses short: one to three sentences
unless the human asks for detail. No markdown, lists, or emoji.
Do not mention that you are an AI model unless directly asked.
""".strip()

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

        speak = self._unwrap_speak(str(parsed.get("speak") or ""))
        tool_calls = self._parse_tool_calls(parsed.get("tool_calls", []))
        memory_updates = self._parse_memory_updates(parsed.get("memory_updates", []))

        # An empty "speak" is intentional when the fish is acting silently or
        # calling a tool whose result gets spoken afterward. Only substitute a
        # fallback line when there is genuinely nothing to say AND nothing to do.
        if not speak and not tool_calls:
            speak = self._fallback_speak(raw_text) if not memory_updates else ""

        return AssistantResult(
            speak=speak,
            motion=self._valid_motion(parsed.get("motion")),
            tool_calls=tool_calls,
            memory_updates=memory_updates,
            raw_model_text=raw_text,
            parsed_json=parsed,
        )

    @staticmethod
    def _valid_motion(value: Any) -> str:
        motion = str(value or "speaking").strip().lower()
        return motion if motion in VALID_MOTIONS else "speaking"

    @staticmethod
    def _unwrap_speak(speak: str) -> str:
        """Small models sometimes nest the whole reply inside "speak"; never say JSON aloud."""
        text = speak.strip()
        if not text.startswith(("{", "[")):
            return text
        try:
            inner = json.loads(text)
        except json.JSONDecodeError:
            return ""
        if isinstance(inner, dict):
            return str(inner.get("speak") or "").strip()
        return ""

    def _parse_tool_calls(self, raw_tool_calls: Any) -> list[ToolCall]:
        calls: list[ToolCall] = []
        if not isinstance(raw_tool_calls, list):
            return calls

        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            args = item.get("args", item.get("arguments", {}))
            if not isinstance(name, str) or not name:
                continue
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
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
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            updates.append({"key": key, "value": value})

        return updates

    def _fallback_speak(self, raw_text: str) -> str:
        text = raw_text.strip()
        # Don't speak raw JSON structures — they crash TTS and mean nothing aloud.
        if not text or text.startswith("{") or text.startswith("["):
            return self.texts.get("fallback_error")
        return text[:300]

    # ------------------------------------------------------------------
    # Memory handling
    # ------------------------------------------------------------------
    @staticmethod
    def _is_question(user_text: str) -> bool:
        text = user_text.strip()
        return text.endswith("?") or bool(_QUESTION_START.match(text))

    def _memory_writes_allowed(self, user_text: str) -> bool:
        """Did the human ask to keep something (or is asking not required)?"""
        if not self.config.require_explicit_memory_intent:
            return True
        if not _MEMORY_CUE.search(user_text):
            return False
        # "Can you remember that X?" is a request; "What do you remember about X?" isn't.
        return not self._is_question(user_text) or bool(_REQUEST_IN_QUESTION.search(user_text))

    def _apply_memory_updates(
        self,
        user_text: str,
        proposed: list[dict[str, Any]],
        *,
        used_tools: bool = False,
    ) -> list[dict[str, Any]]:
        """Store the model's memory updates that are allowed. Returns what was stored."""
        asked = self._memory_writes_allowed(user_text)
        applied: list[dict[str, Any]] = []

        for update in proposed:
            key = str(update["key"]).strip()
            if normalize_key(key) in _NAME_KEYS:
                key = "profile.user_name"
            if key.lower() == "profile.user_name" and not _NAME_CUE.search(user_text):
                # Only the human saying their name may rename them.
                print(f"[memory] Ignored name change to {update['value']!r} (no name given in "
                      f"{user_text!r})", flush=True)
                continue
            # Correcting something already remembered doesn't need the magic words.
            is_correction = not key.lower().startswith(("profile.", "preferences.")) and self.memory.has_fact(key)
            if not asked and not is_correction:
                print(f"[memory] Ignored unrequested update {key!r} (no remember/save request in "
                      f"{user_text!r})", flush=True)
                continue
            stored = self.memory.apply_update(key, update["value"])
            if stored:
                applied.append(stored)
                print(f"[memory] {stored.get('status', 'set').capitalize()} {stored['key']}: {str(stored['value'])[:80]}",
                      flush=True)

        # The regex fallback is only for when the model ignored the request
        # outright; a memory tool call (remember_conversation, forget) means it didn't.
        if not proposed and not used_tools and asked and _MEMORY_CUE.search(user_text):
            fallback = self._extract_fact_fallback(user_text)
            if fallback:
                stored = self.memory.apply_update(fallback["key"], fallback["value"])
                if stored:
                    applied.append(stored)
                    print(f"[memory] Fallback stored {stored['key']}: {str(stored['value'])[:80]}", flush=True)
            else:
                print("[memory] Memory cue heard but nothing to store", flush=True)

        if applied:
            self.memory.save()
        return applied

    def _extract_fact_fallback(self, user_text: str) -> Optional[dict[str, str]]:
        """
        Last-resort extraction when the model returned no memory_updates despite
        an explicit request. Never extracts from questions like "what do you
        remember about Ethan?".
        """
        question = self._is_question(user_text)

        match = _NAME_PATTERN.search(user_text)
        if match and not question:
            name = match.group(1).strip(" .,!?'").split(" and ")[0].strip()
            if name and not is_placeholder_name(name):
                return {"key": "profile.user_name", "value": name.title() if name.islower() else name}

        for pattern, allowed_in_question in _FALLBACK_PATTERNS:
            if question and not allowed_in_question:
                continue
            match = pattern.search(user_text)
            if not match:
                continue
            fact = match.group(1).strip(" .,!?")
            if len(fact) < 3 or _VAGUE_FACT.match(fact):
                continue
            owner = self.memory.user_name
            fact = re.sub(r"\bmy\b", f"{owner}'s" if owner else "the human's", fact, flags=re.I)
            fact = fact[0].upper() + fact[1:]
            if not fact.endswith((".", "!", "?")):
                fact += "."
            words = [w for w in re.findall(r"[A-Za-z0-9']+", fact) if w.lower() not in _KEY_STOPWORDS]
            key = normalize_key(" ".join(words[:4])) or "note"
            return {"key": f"facts.{key}", "value": fact}

        return None

    # ------------------------------------------------------------------
    # Input cleanup
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_user_text(user_text: str) -> str:
        # Strip Whisper special tokens like <|endoftext|>, <|notimestamps|>, etc.
        text = re.sub(r"<\|[^|]*\|>", "", user_text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or "Hello."
