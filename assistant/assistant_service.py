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

from llm_service import LlmService, LlmServiceError


# ----------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------

@dataclass
class AssistantConfig:
    assistant_name: str = "Fishseus"
    user_name: str = "Caleb"

    personality_path: Path = Path("../config/personality_prompt.txt")
    memory_path: Path = Path("../data/assistant_memory.json")
    history_path: Path = Path("../data/conversation_log.jsonl")

    max_history_turns: int = 8
    max_tool_calls: int = 3

    temperature: float = 0.7
    max_tokens: int = 350

    # If true, memory is only saved when the user's command explicitly asks
    # the assistant to remember something.
    require_explicit_memory_intent: bool = True


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
    risk: str = "safe"  # safe | confirm | blocked


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
            f"Personality: {preferences.get('personality', 'dramatic sarcastic fish oracle')}",
        ]

        recent_facts = facts[-6:] if isinstance(facts, list) else []
        for fact in recent_facts:
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
        if not self._tools:
            return "No tools are currently available."

        lines = []
        for tool in self._tools.values():
            if tool.risk == "safe":
                lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def execute(self, call: ToolCall) -> dict[str, Any]:
        tool = self._tools.get(call.name)
        if tool is None:
            return {
                "tool": call.name,
                "ok": False,
                "error": "Unknown tool",
            }

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
            }
        except Exception as exc:
            return {
                "tool": call.name,
                "ok": False,
                "error": str(exc),
            }


# ----------------------------------------------------------------------
# Assistant service
# ----------------------------------------------------------------------

class AssistantService:
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

    def handle_user_text(self, user_text: str) -> AssistantResult:
        """
        Main entrypoint for the orchestrator.

        user_text should already have the wake word stripped by stt_service or
        the orchestrator, but this method does not require that.
        """
        clean_user_text = self._clean_user_text(user_text)
        self.memory.update_path("session.last_command", clean_user_text)

        # Deterministic local memory capture before the LLM call.
        local_memory_updates = self._detect_explicit_memory_updates(clean_user_text)
        for update in local_memory_updates:
            self._apply_memory_update(update)

        messages = self._build_messages(clean_user_text)

        start = time.monotonic()
        try:
            llm_result = self.llm.chat(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                # Ollama may or may not support this. If unsupported, it is
                # ignored or can be removed in llm_service.py config.
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

        # Merge deterministic memory updates with model-suggested updates.
        result.memory_updates = [*local_memory_updates, *result.memory_updates]

        # Only apply model-suggested memory if explicit memory intent exists.
        if self._memory_writes_allowed(clean_user_text):
            for update in result.memory_updates:
                self._apply_memory_update(update)
            self.memory.save()
        elif local_memory_updates:
            # Explicit local detector found memory intent, so save those.
            self.memory.save()

        # Execute safe tools requested by model.
        result.tool_calls = result.tool_calls[: self.config.max_tool_calls]
        result.tool_results = [self.tool_registry.execute(call) for call in result.tool_calls]

        self._append_history("user", clean_user_text)
        self._append_history("assistant", result.speak)
        self._log_turn(clean_user_text, result)

        return result

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_messages(self, user_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._personality_prompt()},
            {"role": "system", "content": self._memory_prompt()},
            {"role": "system", "content": self._tool_prompt()},
            *self.history[-self.config.max_history_turns :],
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
        if not speak:
            speak = self._fallback_speak(raw_text)

        motion = str(parsed.get("motion") or "speaking").strip().lower()
        if motion not in {"idle", "speaking", "happy", "annoyed", "thinking", "excited"}:
            motion = "speaking"

        tool_calls = self._parse_tool_calls(parsed.get("tool_calls", []))
        memory_updates = self._parse_memory_updates(parsed.get("memory_updates", []))

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
        if not text:
            return "My thoughts got tangled in the kelp. Try that again."
        # If JSON parsing failed, speak a trimmed raw response rather than dying.
        return text[:300]

    # ------------------------------------------------------------------
    # Memory handling
    # ------------------------------------------------------------------
    def _memory_writes_allowed(self, user_text: str) -> bool:
        if not self.config.require_explicit_memory_intent:
            return True
        return bool(
            re.search(
                r"\b(remember that|remember this|from now on|call me|my name is|i prefer|my preference is)\b",
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

        # Facts are append-only so we don't overwrite earlier notes.
        if key.startswith("facts."):
            fact_key = key.split(".", 1)[1] if "." in key else "note"
            self.memory.add_fact(fact_key, str(value))
        else:
            self.memory.update_path(key, value)

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
        text = re.sub(r"\s+", " ", user_text).strip()
        return text or "Hello."


# ----------------------------------------------------------------------
# Demo tools and manual test
# ----------------------------------------------------------------------

def build_demo_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def wiggle(cycles: int = 1) -> str:
        cycles = max(1, min(int(cycles), 5))
        print(f"[DEMO TOOL] Would wiggle {cycles} cycle(s)")
        return f"wiggled {cycles} cycle(s)"

    def open_mouth() -> str:
        print("[DEMO TOOL] Would open mouth")
        return "opened mouth"

    def set_mode(mode: str) -> str:
        allowed = {"assistant", "bluetooth"}
        if mode not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}")
        print(f"[DEMO TOOL] Would set mode to {mode}")
        return f"mode set to {mode}"

    def get_current_time() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    registry.register(
        Tool(
            name="wiggle",
            description="Make the fish wiggle. Args: cycles integer from 1 to 5.",
            function=wiggle,
            risk="safe",
        )
    )
    registry.register(
        Tool(
            name="open_mouth",
            description="Open the fish mouth once. Args: none.",
            function=open_mouth,
            risk="safe",
        )
    )
    registry.register(
        Tool(
            name="set_mode",
            description="Switch mode. Args: mode is 'assistant' or 'bluetooth'.",
            function=set_mode,
            risk="safe",
        )
    )
    registry.register(
        Tool(
            name="get_current_time",
            description="Get the current system time. Args: none.",
            function=get_current_time,
            risk="safe",
        )
    )

    return registry


if __name__ == "__main__":
    from llm_service import LlmConfig

    llm = LlmService(
        LlmConfig(
            endpoint_url="http://ollama.angelfish-gamma.ts.net/v1/chat/completions",
            model="gemma4:e2b",
            temperature=0.7,
            max_tokens=350,
            disable_reasoning=True,
        )
    )

    assistant = AssistantService(
        llm=llm,
        config=AssistantConfig(
            memory_path=Path("../data/assistant_memory.json"),
            history_path=Path("../data/conversation_log.jsonl"),
        ),
        tool_registry=build_demo_tool_registry(),
    )

    try:
        print("Fishseus assistant demo. Type 'exit' to quit.")
        while True:
            user_text = input("You: ").strip()
            if user_text.lower() in {"exit", "quit"}:
                break

            result = assistant.handle_user_text(user_text)
            print("\n--- AssistantResult ---")
            print(f"Speak:       {result.speak}")
            print(f"Motion:      {result.motion}")
            print(f"Tool calls:  {[call.__dict__ for call in result.tool_calls]}")
            print(f"Tool results:{result.tool_results}")
            print(f"Memory:      {result.memory_updates}")
            print(f"Elapsed:     {result.elapsed_s:.2f}s")
            print()
    finally:
        llm.close()
