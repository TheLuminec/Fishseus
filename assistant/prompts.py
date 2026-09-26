"""
prompts.py — Editable prompt text for the assistant.

The personality (character and voice) lives in config/personality_prompt.txt.
Everything else the assistant says *to the model* lives here: the response
rules, the follow-up nudge after tool results, the sensor-event framing, and
the fallback lines spoken when things go wrong.

Defaults are defined in code. Edits from the web UI are stored as overrides in
config/prompt_texts.json (only keys that differ from the default), so a code
update can improve a default without clobbering customised text.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


DEFAULT_RULES = """
HOW TO REPLY
Reply with ONLY one JSON object in exactly this shape. No text outside it, no markdown, no code fences:
{"speak": "what to say aloud, or an empty string", "motion": "speaking", "tool_calls": [], "memory_updates": []}

- "speak" is read aloud. It may be "" only when you are calling a tool or doing an action.
- "motion" is your body language, one of:
  happy (compliments, jokes, good news), excited (surprises, being asked to perform),
  thinking (hard questions, lookups, math), annoyed (things you can't do, failures, repeated questions),
  speaking (neutral answers, the default), idle (the human wants quiet or is leaving).
- "tool_calls" is [] unless you need a tool. Format: [{"name": "tool_name", "args": {"arg": "value"}}]
- "memory_updates" is [] unless the human asked you to remember something (see MEMORY).

TOOLS
The tools you can use right now are listed under AVAILABLE TOOLS, each with an example. Only call tools from that list.
- To DO something (move, play music, change voice), call the tool. Saying you did it is not doing it.
- To LOOK SOMETHING UP (weather, the camera, a calculation), call the tool and set "speak" to "". The result comes back to you and you answer then, in one clean reply. You may put a short line in "speak" while you check, but only when it earns its place.
- Never read raw data aloud like a machine. Fold it into a sentence in your own voice.
- Don't call a tool for something you already know, and never just to be calling one.

MEMORY
LONG-TERM MEMORY below is everything you know long-term; answer from it directly. The recent conversation is the chat itself, with times.
Write to long-term memory only when the human asks you to remember, save, or note something, tells you their name, or tells you how to respond:
  "memory_updates": [{"key": "facts.ethan_address", "value": "Ethan Weber lives at 5450 Lockwood Road, Madison, Ohio."}]
- Facts, addresses, dates, people, anything else: "facts." plus a short snake_case label. The value is one complete sentence that will make sense on its own months from now. Say who it's about ("Caleb's favorite food is sausage.", not "sausage").
- To correct or add to something you already remember, reuse that fact's [label] and write the full corrected sentence. It replaces the old one.
- The human's own name: "profile.user_name". How they want you to respond: "preferences.response_style".
- To delete one remembered fact, call the forget tool with its label.
- To keep the whole recent conversation ("remember this conversation"), call the remember_conversation tool instead.
- Only say you've remembered something when you actually put it in "memory_updates" or called a memory tool.

EXAMPLES
User: "tell me a fish joke"
{"speak": "Why don't fish play piano? You cannot trust something that swims toward its own scales.", "motion": "happy", "tool_calls": [], "memory_updates": []}

User: "call me Caleb"
{"speak": "Caleb it is. A name the tide will not wash off.", "motion": "happy", "tool_calls": [], "memory_updates": [{"key": "profile.user_name", "value": "Caleb"}]}

User: "remember that Ethan's address is 5450 Lockwood Road, Madison Ohio"
{"speak": "Filed away where the water cannot reach it.", "motion": "speaking", "tool_calls": [], "memory_updates": [{"key": "facts.ethan_address", "value": "Ethan's address is 5450 Lockwood Road, Madison, Ohio."}]}

User: "oh, and his zip code is 44057"
{"speak": "Zip code tucked in with the rest of it.", "motion": "speaking", "tool_calls": [], "memory_updates": [{"key": "facts.ethan_address", "value": "Ethan's address is 5450 Lockwood Road, Madison, Ohio 44057."}]}
""".strip()


DEFAULT_TOOL_FOLLOWUP = """
Those are the results of your tool calls. Now answer the human's request in your own voice, folding the information in naturally. Don't read raw data verbatim or repeat a line you already said. If a tool failed, say so plainly, with style, instead of pretending it worked. Reply with the same JSON shape, with "tool_calls": [] and "memory_updates": [].
""".strip()


DEFAULT_SENSOR_EVENT = """
That was a sensor event, not speech. React briefly in character (one or two sentences): greet, comment, or stay quiet. Use the recent conversation if it's relevant. Don't call tools or write memory. If no reaction is warranted, set "speak" to "".
""".strip()


DEFAULT_CONVERSATION_SUMMARY = """
You keep the long-term memory of Fishseus, a talking fish assistant. Fishseus only recalls the recent conversation for a limited time, so anything worth keeping has to be written to long-term memory now.

Read the conversation and decide what is worth remembering for months: facts about the people in it (names, relationships, birthdays, addresses, preferences, plans, promises), decisions that were made, things the human wants done later, and anything the human called important. Skip greetings, small talk, jokes, and routine tool use (weather, math, music) unless something lasting came out of it. Skip anything already in long-term memory unless it changed.

Reply with ONLY a JSON object:
{"summary": "One or two sentences on what the conversation covered, written to make sense months from now.", "facts": [{"key": "short_snake_case_label", "value": "One complete sentence that makes sense on its own."}]}
- To update something already in long-term memory, reuse its label and write the full corrected sentence.
- Say who each fact is about ("Caleb's sister Maya's birthday is May 7.", not "Her birthday is May 7.").
- "facts" may be empty when nothing new is worth keeping.
""".strip()


DEFAULTS: dict[str, str] = {
    "rules": DEFAULT_RULES,
    "tool_followup": DEFAULT_TOOL_FOLLOWUP,
    "sensor_event": DEFAULT_SENSOR_EVENT,
    "conversation_summary": DEFAULT_CONVERSATION_SUMMARY,
    "memory_saved": "Got it. That's tucked away where the tide can't reach it.",
    "fallback_error": "My thoughts got tangled in the kelp. Try that again.",
    "fallback_offline": "My thinking currents have gone still for a moment. Give me a breath and try again.",
}

# (label, help) shown in the web UI, in display order.
DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "rules": (
        "Response rules",
        "Sent after the personality every turn: the JSON reply format, how to use tools, how to "
        "write memory, plus a few core examples. Tool-specific examples come from each tool.",
    ),
    "tool_followup": (
        "After tool results",
        "Sent with tool results (weather, camera, math…) to get the fish's spoken answer.",
    ),
    "sensor_event": (
        "Sensor events",
        "Framing for motion/door sensor triggers, so the fish reacts instead of treating them as speech.",
    ),
    "conversation_summary": (
        "Remembering a conversation",
        "Instructions for the remember_conversation tool: how to boil the recent conversation down "
        "to a summary and lasting facts for long-term memory.",
    ),
    "memory_saved": (
        "Spoken after a silent save",
        "Said when the model stored a memory but forgot to say anything about it.",
    ),
    "fallback_error": (
        "Spoken when the reply is garbled",
        "Said aloud when the model's reply can't be understood.",
    ),
    "fallback_offline": (
        "Spoken when the model is unreachable",
        "Said aloud when the LLM request fails or times out.",
    ),
}


class PromptTexts:
    """Default prompt text with persistent per-key overrides."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._overrides: dict[str, str] = {}
        self._mtime: float | None = None
        self.load()

    def load(self) -> None:
        with self._lock:
            self._overrides = {}
            self._mtime = None
            if not self.path.exists():
                return
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._mtime = self.path.stat().st_mtime
            except Exception as exc:
                print(f"[prompts] Could not read {self.path.name}, using defaults: {exc}", flush=True)
                return
            if isinstance(data, dict):
                self._overrides = {k: v for k, v in data.items() if k in DEFAULTS and isinstance(v, str)}

    def _reload_if_changed(self) -> None:
        """Pick up hand edits to the overrides file without a restart."""
        try:
            mtime = self.path.stat().st_mtime if self.path.exists() else None
        except OSError:
            return
        if mtime != self._mtime:
            self.load()

    def get(self, key: str) -> str:
        with self._lock:
            self._reload_if_changed()
            return self._overrides.get(key, DEFAULTS[key])

    def set(self, key: str, text: str) -> None:
        if key not in DEFAULTS:
            raise KeyError(key)
        text = str(text or "").strip()
        with self._lock:
            if not text or text == DEFAULTS[key]:
                self._overrides.pop(key, None)
            else:
                self._overrides[key] = text
            self._save()

    def reset(self, key: str) -> None:
        if key not in DEFAULTS:
            raise KeyError(key)
        with self._lock:
            self._overrides.pop(key, None)
            self._save()

    def describe(self) -> list[dict[str, Any]]:
        with self._lock:
            self._reload_if_changed()
            return [
                {
                    "key": key,
                    "label": label,
                    "help": help_text,
                    "text": self._overrides.get(key, DEFAULTS[key]),
                    "default": DEFAULTS[key],
                    "customized": key in self._overrides,
                }
                for key, (label, help_text) in DESCRIPTIONS.items()
            ]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._overrides:
            if self.path.exists():
                self.path.unlink()
            self._mtime = None
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._overrides, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)
        self._mtime = self.path.stat().st_mtime
