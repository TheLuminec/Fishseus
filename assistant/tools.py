"""
tools.py — Tool definitions for the Fishseus assistant.

All tool implementations live here so the orchestrator, web server, and
tests can share a single source of truth.  The registry is built via
build_tool_registry(), which accepts getter callables for each service so
the tools always reference the live instance even if services restart.

To add a new tool:
  1. Write the function below.
  2. Add a Tool(...) entry in the list at the bottom of build_tool_registry.
"""

from __future__ import annotations

import ast
import random
import time
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Callable, Optional

from assistant_service import Tool, ToolRegistry

if TYPE_CHECKING:
    from assistant_service import AssistantService
    from motion_service import MotionService
    from tts_service import TtsService


def build_tool_registry(
    get_motion: Callable[[], Optional["MotionService"]],
    get_tts: Callable[[], Optional["TtsService"]],
    get_assistant: Callable[[], Optional["AssistantService"]],
) -> ToolRegistry:
    """
    Instantiate and populate the tool registry.

    Parameters are zero-argument callables so tools always resolve the
    current live service instance at call time, not at build time.
    """
    registry = ToolRegistry()

    # ------------------------------------------------------------------
    # Action tools  (returns_data=False — result is never spoken)
    # ------------------------------------------------------------------

    def wiggle(cycles: int = 1) -> str:
        motion = get_motion()
        if motion is None:
            return "motion unavailable"
        cycles = max(1, min(int(cycles), 5))
        motion.wiggle(cycles=cycles)
        return f"wiggle queued for {cycles} cycle(s)"

    def open_mouth() -> str:
        motion = get_motion()
        if motion is None:
            return "motion unavailable"
        motion.open_mouth()
        return "mouth open queued"

    def set_mode(mode: str) -> str:
        allowed = {"assistant", "bluetooth"}
        if mode not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}")
        return f"mode switch requested: {mode}"

    def set_voice(voice: str) -> str:
        tts = get_tts()
        if tts is None:
            return "tts unavailable"
        tts.set_voice(voice)
        return f"voice set to {voice}"

    # ------------------------------------------------------------------
    # Raw data tools  (returns_data=True, synthesize_result=False)
    # Short values the orchestrator can speak directly without an LLM pass.
    # ------------------------------------------------------------------

    def get_current_time() -> str:
        return time.strftime("%I:%M %p")

    def get_date() -> str:
        return time.strftime("%A, %B %d, %Y")

    def flip_coin() -> str:
        return random.choice(["heads", "tails"])

    def roll_dice(sides: int = 6, count: int = 1) -> str:
        sides = max(2, min(int(sides), 100))
        count = max(1, min(int(count), 10))
        rolls = [random.randint(1, sides) for _ in range(count)]
        if count == 1:
            return str(rolls[0])
        total = sum(rolls)
        return f"{', '.join(str(r) for r in rolls)}, total {total}"

    def calculate(expression: str) -> str:
        _SAFE = {
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
            ast.Mod, ast.Pow, ast.USub, ast.UAdd,
        }
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            for node in ast.walk(tree):
                if type(node) not in _SAFE:
                    return "Only basic arithmetic is supported"
            result = eval(compile(tree, "<expr>", "eval"))  # noqa: S307 — guarded above
            if isinstance(result, float):
                return str(int(result)) if result.is_integer() else f"{result:.6g}"
            return str(result)
        except ZeroDivisionError:
            return "Division by zero — even the ocean has limits"
        except Exception:
            return "Could not parse that expression"

    def list_voices() -> str:
        tts = get_tts()
        if tts is None:
            return "tts unavailable"
        voices = tts.available_voices()
        return ", ".join(voices) if voices else "no voices found"

    # ------------------------------------------------------------------
    # Synthesised data tools  (returns_data=True, synthesize_result=True)
    # Richer data that the orchestrator feeds back to the LLM for a
    # natural spoken response before playing audio.
    # ------------------------------------------------------------------

    def get_weather(location: str = "") -> str:
        try:
            encoded = urllib.parse.quote(location.strip())
            url = (
                f"https://wttr.in/{encoded}?format=%l:+%C,+%t"
                if encoded else
                "https://wttr.in/?format=%l:+%C,+%t"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Fishseus/1.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                text = resp.read().decode("utf-8", errors="ignore").strip()
            text = text.encode("ascii", "ignore").decode().strip()
            return text or "Weather data unavailable"
        except Exception as exc:
            return f"Could not reach the weather currents: {exc}"

    def recall_memory() -> str:
        asst = get_assistant()
        if asst is None:
            return "Memory not initialised yet"
        return asst.memory.compact_summary()

    # ------------------------------------------------------------------
    # Tool table
    # ------------------------------------------------------------------
    _tools = [
        #  name               description                                                               fn                risk     returns_data  synthesize_result
        Tool("wiggle",           "Make the fish wiggle. Args: cycles int 1-5.",                           wiggle,           "safe",  False,        False),
        Tool("open_mouth",       "Open the fish mouth once. No args.",                                    open_mouth,       "safe",  False,        False),
        Tool("set_mode",         "Request a mode switch. Args: mode — 'assistant' or 'bluetooth'.",       set_mode,         "safe",  False,        False),
        Tool("get_current_time", "Get the current clock time. No args.",                                  get_current_time, "safe",  True,         False),
        Tool("get_date",         "Get today's full date. No args.",                                       get_date,         "safe",  True,         False),
        Tool("get_weather",      "Get current weather. Args: location string (city or empty for local).", get_weather,      "safe",  True,         True),
        Tool("flip_coin",        "Flip a coin. Returns heads or tails. No args.",                         flip_coin,        "safe",  True,         False),
        Tool("roll_dice",        "Roll dice. Args: sides int (default 6), count int (default 1).",        roll_dice,        "safe",  True,         False),
        Tool("calculate",        "Evaluate a math expression. Args: expression string.",                  calculate,        "safe",  True,         False),
        Tool("recall_memory",    "Read everything remembered about the user. No args.",                   recall_memory,    "safe",  True,         True),
        Tool("list_voices",      "List available Piper TTS voices. No args.",                             list_voices,      "safe",  True,         False),
        Tool("set_voice",        "Set the TTS voice. Args: voice name string.",                           set_voice,        "safe",  False,        False),
    ]
    for t in _tools:
        registry.register(t)

    return registry
