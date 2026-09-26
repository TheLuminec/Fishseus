"""
tools.py — Tool definitions for the Fishseus assistant.

All tool implementations live here so the orchestrator, web server, and
tests can share a single source of truth.  The registry is built via
build_tool_registry(), which accepts getter callables for each service so
the tools always reference the live instance even if services restart.

Each tool carries its own prompt guidance: a description (what it does and its
args), a hint (when to use it), and examples (what the user says -> the call).
Only tools that are enabled and whose service is running are shown to the
model, examples included. All of it is editable from the web UI's Tools page;
edits are saved to config/tool_overrides.json.

To add a new tool:
  1. Write the function below.
  2. Add a Tool(...) entry in the list at the bottom of build_tool_registry,
     with a hint and at least one example.
"""

from __future__ import annotations

import ast
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from assistant.assistant_service import Tool, ToolRegistry
from services import ROOT_DIR

if TYPE_CHECKING:
    from access_point.access_point_service import AccessPointService
    from assistant.assistant_service import AssistantService
    from bluetooth.bluetooth_service import BluetoothService
    from motion.motion_service import MotionService
    from sensors.sensor_service import SensorService
    from spotify.spotify_service import SpotifyService
    from tts.tts_service import TtsService
    from vision.vision_service import VisionService


DEFAULT_OVERRIDES_PATH = ROOT_DIR / "config" / "tool_overrides.json"


def _none() -> None:
    return None


def build_tool_registry(
    get_motion: Callable[[], Optional["MotionService"]],
    get_tts: Callable[[], Optional["TtsService"]],
    get_assistant: Callable[[], Optional["AssistantService"]],
    get_vision: Callable[[], Optional["VisionService"]] = _none,
    get_sensors: Callable[[], Optional["SensorService"]] = _none,
    get_bluetooth: Callable[[], Optional["BluetoothService"]] = _none,
    get_spotify: Callable[[], Optional["SpotifyService"]] = _none,
    get_access_point: Callable[[], Optional["AccessPointService"]] = _none,
    overrides_path: Optional[Path] = DEFAULT_OVERRIDES_PATH,
) -> ToolRegistry:
    """
    Instantiate and populate the tool registry.

    Parameters are zero-argument callables so tools always resolve the
    current live service instance at call time, not at build time.
    """
    registry = ToolRegistry(overrides_path)

    def running(getter: Callable[[], object]) -> Callable[[], bool]:
        return lambda: getter() is not None

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

    def clear_session() -> str:
        asst = get_assistant()
        if asst is None:
            return "assistant not available"
        asst.clear_history()
        return "conversation history cleared"

    def remember_conversation(focus: str = "") -> str:
        asst = get_assistant()
        if asst is None:
            return "Memory not available"
        return asst.remember_conversation(focus)

    def forget(key: str = "") -> str:
        asst = get_assistant()
        if asst is None:
            return "Memory not available"
        query = (key or "").strip()
        if not query:
            return "Nothing specified to forget"
        removed = asst.memory.forget(query)
        if not removed:
            return f"No remembered fact matched '{query}', so nothing was deleted"
        asst.memory.save()
        labels = ", ".join(f"{r['key']} ({r['value']})" for r in removed)
        print(f"[memory] Forgot {len(removed)} fact(s): {labels}", flush=True)
        if len(removed) == 1:
            return f"Deleted the memory: {removed[0]['value']}"
        return f"Deleted {len(removed)} memories: " + "; ".join(r["value"] for r in removed)

    def look(camera: str = "", question: str = "") -> str:
        vision = get_vision()
        if vision is None:
            return "My eyes are not connected — no camera available"
        prompt = question.strip() or "Describe what you see in one or two sentences."
        try:
            print(f"[vision] Capturing from camera '{camera or 'default'}'…", flush=True)
            result = vision.look(camera, prompt)
            print(f"[vision] -> {result[:100]}", flush=True)
            return result
        except Exception as exc:
            return f"My vision is clouded: {exc}"

    def check_sensors() -> str:
        sensors = get_sensors()
        if sensors is None:
            return "No sensors are connected"
        report = sensors.sensor_report()
        if not report:
            return "No sensors are configured"
        lines = []
        for s in report:
            since = s.get("seconds_since_trigger")
            if since is None:
                lines.append(f"{s['name']}: never triggered")
            else:
                lines.append(f"{s['name']}: last triggered {int(since)} seconds ago")
        return "; ".join(lines)

    # ------------------------------------------------------------------
    # Music & speaker tools  (returns_data=True, synthesize_result=False)
    # Each returns a short confirmation or a readable failure; the fish
    # speaks it as-is so a missing device is explained, not swallowed.
    # ------------------------------------------------------------------

    def _spotify_call(action: Callable[["SpotifyService"], str]) -> str:
        spotify = get_spotify()
        if spotify is None:
            return "Spotify is not connected"
        try:
            return action(spotify)
        except Exception as exc:
            return f"Spotify hiccup: {exc}"

    def play_music(query: str = "", kind: str = "track") -> str:
        return _spotify_call(lambda sp: sp.play(query, kind))

    def pause_music() -> str:
        return _spotify_call(lambda sp: sp.pause())

    def resume_music() -> str:
        return _spotify_call(lambda sp: sp.resume())

    def skip_track() -> str:
        return _spotify_call(lambda sp: sp.next_track())

    def previous_track() -> str:
        return _spotify_call(lambda sp: sp.previous_track())

    def set_music_volume(percent: int = 50) -> str:
        return _spotify_call(lambda sp: sp.set_volume(percent))

    def now_playing() -> str:
        return _spotify_call(lambda sp: sp.now_playing())

    def connect_speaker(name: str = "") -> str:
        bt = get_bluetooth()
        if bt is None:
            return "Bluetooth is not available"
        try:
            device = bt.connect(name)
            return f"Connected to {device['name']}"
        except Exception as exc:
            return f"Couldn't connect the speaker: {exc}"

    def pair_speaker(name: str = "") -> str:
        bt = get_bluetooth()
        if bt is None:
            return "Bluetooth is not available"
        if not name.strip():
            return "Tell me the speaker's name to pair with"
        try:
            device = bt.pair(name)
            return f"Paired and connected to {device['name']}"
        except Exception as exc:
            return f"Couldn't pair the speaker: {exc}"

    def disconnect_speaker(name: str = "") -> str:
        bt = get_bluetooth()
        if bt is None:
            return "Bluetooth is not available"
        try:
            names = bt.disconnect(name)
            return f"Disconnected {', '.join(names)}" if names else "No speaker was connected"
        except Exception as exc:
            return f"Couldn't disconnect: {exc}"

    def network_clients() -> str:
        ap = get_access_point()
        if ap is None:
            return "The Fishseus Wi-Fi network is not running"
        try:
            clients = ap.clients()
        except Exception as exc:
            return f"Couldn't check the network: {exc}"
        if not clients:
            return "No devices are connected to the Fishseus network"
        labels = [c["hostname"] or c["ip"] or c["mac"] for c in clients]
        return f"{len(clients)} connected: {', '.join(labels)}"

    # ------------------------------------------------------------------
    # Tool table
    # ------------------------------------------------------------------
    has_motion = running(get_motion)
    has_tts = running(get_tts)
    has_assistant = running(get_assistant)
    has_spotify = running(get_spotify)
    has_bluetooth = running(get_bluetooth)

    def ex(user: str, args: Optional[dict] = None, **extra: str) -> dict:
        return {"user": user, "args": args or {}, **extra}

    _tools = [
        # --- Body ---
        Tool("wiggle", "Wiggle your body and tail. Args: cycles (int 1-5).",
             wiggle, "safe", False, False,
             hint="Whenever you're asked to move, dance, or perform, or for a burst of enthusiasm. "
                  "Usually do it silently.",
             examples=[ex("wiggle for me", {"cycles": 2}, motion="excited"),
                       ex("do a little dance", {"cycles": 4}, motion="excited")],
             available=has_motion),
        Tool("open_mouth", "Open your mouth once. No args.",
             open_mouth, "safe", False, False,
             examples=[ex("open your mouth")],
             available=has_motion),
        Tool("set_mode", "Request a mode switch. Args: mode ('assistant' or 'bluetooth').",
             set_mode, "safe", False, False,
             hint="Only when explicitly asked to switch modes."),

        # --- Quick answers ---
        # The current date and time are already in every prompt, so these are
        # off by default; enable them from the web UI if you want tool calls.
        Tool("get_current_time", "Get the current clock time. No args.",
             get_current_time, "safe", True, False, enabled=False,
             examples=[ex("what time is it?")]),
        Tool("get_date", "Get today's full date. No args.",
             get_date, "safe", True, False, enabled=False,
             examples=[ex("what's the date today?")]),
        Tool("flip_coin", "Flip a coin. Returns heads or tails. No args.",
             flip_coin, "safe", True, False,
             hint="Decisions, games, or whenever a little chance is fun.",
             examples=[ex("flip a coin")]),
        Tool("roll_dice", "Roll dice. Args: sides (int, default 6), count (int, default 1).",
             roll_dice, "safe", True, False,
             examples=[ex("roll a d20", {"sides": 20}),
                       ex("roll two dice", {"count": 2})]),
        Tool("calculate", "Evaluate an arithmetic expression. Args: expression (string, Python syntax).",
             calculate, "safe", True, False,
             hint="Any arithmetic at all. Never do the math in your head; pass the expression.",
             examples=[ex("what is 47 times 83?", {"expression": "47 * 83"}),
                       ex("what's 15 percent of 240?", {"expression": "240 * 0.15"})]),

        # --- The world ---
        Tool("get_weather", "Get current weather. Args: location (city, or empty for right here).",
             get_weather, "safe", True, True,
             hint="Weather, temperature, or what to wear. Leave location empty for here.",
             examples=[ex("what's the weather like?"),
                       ex("is it raining in Chicago?", {"location": "Chicago"})]),
        Tool("look", "Look through the camera and describe what's there. Args: camera (optional name), "
                     "question (optional, what to look for).",
             look, "safe", True, True,
             hint="What you can see, who is there, anything about the room. Pass a question to look "
                  "for something specific.",
             examples=[ex("what do you see?"),
                       ex("is anyone in the kitchen?",
                          {"question": "Is a person visible? Describe who they are and what they are doing."})],
             available=running(get_vision)),
        Tool("check_sensors", "Check when each sensor (motion detector etc.) last triggered. No args.",
             check_sensors, "safe", True, False,
             hint="Recent movement or activity nearby.",
             examples=[ex("has anyone walked by lately?")],
             available=running(get_sensors)),

        # --- Memory & conversation ---
        Tool("remember_conversation", "Summarize the recent conversation and save what matters to long-term "
                                      "memory, so it outlasts your short recall. Args: focus (optional, anything "
                                      "the human especially wants kept).",
             remember_conversation, "safe", True, True,
             hint="When asked to remember this conversation, what you talked about, or everything from today. "
                  "For one specific fact, use memory_updates instead.",
             examples=[ex("remember this conversation"),
                       ex("save what we talked about, especially the party plans", {"focus": "the party plans"})],
             available=has_assistant),
        Tool("forget", "Delete one fact from long-term memory. Args: key (the fact's [label]).",
             forget, "safe", True, True,
             hint="When asked to forget or delete one specific thing you remember. Use the fact's "
                  "[label] from LONG-TERM MEMORY. This erases it permanently and is not clear_session.",
             examples=[ex("forget Ethan's address", {"key": "ethan_address"})],
             available=has_assistant),
        Tool("clear_session", "Forget the recent conversation for a fresh start. Long-term memory stays. No args.",
             clear_session, "safe", False, False,
             hint="When asked to start over or forget what you've been talking about. To drop one "
                  "remembered fact, use forget instead.",
             examples=[ex("let's start fresh", speak="Fresh water, clean slate.")],
             available=has_assistant),

        # --- Voice ---
        Tool("list_voices", "List available Piper TTS voices. No args.",
             list_voices, "safe", True, False,
             examples=[ex("what voices can you do?")],
             available=has_tts),
        Tool("set_voice", "Set the TTS voice. Args: voice (name from list_voices).",
             set_voice, "safe", False, False,
             examples=[ex("switch to the arctic voice", {"voice": "en_US-arctic-medium"})],
             available=has_tts),

        # --- Music ---
        Tool("play_music", "Play music on Spotify. Args: query (song, artist, album or playlist name; include "
                           "the artist if said), kind ('track' default, 'artist', 'album' or 'playlist').",
             play_music, "safe", True, False,
             examples=[ex("play Bohemian Rhapsody by Queen", {"query": "Bohemian Rhapsody Queen", "kind": "track"}),
                       ex("put on some Miles Davis", {"query": "Miles Davis", "kind": "artist"})],
             available=has_spotify),
        Tool("pause_music", "Pause the Spotify music. No args.",
             pause_music, "safe", True, False,
             examples=[ex("pause the music")],
             available=has_spotify),
        Tool("resume_music", "Resume paused Spotify music. No args.",
             resume_music, "safe", True, False,
             examples=[ex("keep playing")],
             available=has_spotify),
        Tool("skip_track", "Skip to the next song. No args.",
             skip_track, "safe", True, False,
             examples=[ex("skip this song")],
             available=has_spotify),
        Tool("previous_track", "Go back to the previous song. No args.",
             previous_track, "safe", True, False,
             examples=[ex("play the last song again")],
             available=has_spotify),
        Tool("set_music_volume", "Set the music volume. Args: percent (int 0-100).",
             set_music_volume, "safe", True, False,
             examples=[ex("turn the music down to thirty percent", {"percent": 30})],
             available=has_spotify),
        Tool("now_playing", "Say which song is playing. No args.",
             now_playing, "safe", True, False,
             examples=[ex("what song is this?")],
             available=has_spotify),

        # --- Speakers & network ---
        Tool("connect_speaker", "Connect an already-paired Bluetooth speaker. Args: name (optional; empty = default speaker).",
             connect_speaker, "safe", True, False,
             examples=[ex("connect to the speaker")],
             available=has_bluetooth),
        Tool("pair_speaker", "Scan for and pair a new Bluetooth speaker in pairing mode. Args: name.",
             pair_speaker, "safe", True, False,
             examples=[ex("pair with my JBL Flip", {"name": "JBL Flip"})],
             available=has_bluetooth),
        Tool("disconnect_speaker", "Disconnect the Bluetooth speaker. Args: name (optional).",
             disconnect_speaker, "safe", True, False,
             examples=[ex("disconnect the speaker")],
             available=has_bluetooth),
        Tool("network_clients", "List devices connected to the Fishseus Wi-Fi network (turret, camera). No args.",
             network_clients, "safe", True, False,
             examples=[ex("what's connected to your wifi?")],
             available=running(get_access_point)),
    ]
    for t in _tools:
        registry.register(t)

    return registry
