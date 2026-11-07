"""
claude_link_service.py

Two-way link between the fish and Claude (Claude Code on this Pi).

Responsibilities:
- ask(): hand a spoken request to a headless `claude -p` run in the background
  and deliver its short spoken reply back to the orchestrator when it arrives.
  That Claude run can answer directly or relay the message to other Claude
  sessions (the Fishseus Agent, peers) with ListAgents/SendMessage.
- Outbox: watch a directory for *.txt files and hand their text to the
  orchestrator to speak. This is how Claude sessions make the fish talk:
      echo "Dinner is ready" > ~/Docs/fish/say/$(date +%s).txt
  (or use bin/fishsay).
- Log every exchange to a JSONL file so the Fishseus Agent can read them later.

Non-responsibilities:
- No audio, TTS or motion: replies go out through the on_message callback and
  the orchestrator speaks them when the speaker is free.

Example:
    svc = ClaudeLinkService(ClaudeLinkConfig())
    svc.on_message = lambda text, source: print(source, text)
    svc.initialize()
    svc.ask("What's the weather like on Mars?")
    ...
    svc.shutdown()
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from services import Service, ServiceConfig, ServiceError

HOME = Path.home()

LIAISON_PROMPT = """\
You are the Claude link inside Fishseus, a talking Billy Bass fish on a Raspberry Pi.
Someone just spoke to the fish and it passed their words to you. Your reply is read
aloud by the fish's text-to-speech, so:
- Reply in plain spoken English: one to three short sentences, no markdown, lists,
  code, URLs or emoji. Stay under about 60 words.
- Answer directly when you can. For research questions, keep it brief.
- If they want a message passed to another Claude agent, the "Fishseus Agent", or
  the owner, call ListAgents. Then SendMessage to the right session, using the
  name exactly as listed and starting with "Relayed from the fish:". If no
  suitable session is listed, say the message is saved for the Fishseus Agent.
  Every exchange is logged to {log} anyway.
- Background on the project and the house rules is in {docs}. Read it only when
  the request needs it.
- You cannot change files or run commands. If they ask for that, say you've passed
  it on to the Fishseus Agent and send it there.
Recent exchanges (oldest first) for context:
{history}
"""


class ClaudeLinkError(ServiceError):
    pass


@dataclass(frozen=True)
class ClaudeLinkConfig(ServiceConfig):
    module_name: str = "claude_link"

    claude_binary: str = "claude"            # resolved on PATH, then ~/.local/bin
    model: str = ""                          # "" = account default
    workdir: Path = HOME / "Docs" / "fish" / "liaison"
    docs_path: Path = HOME / "Docs" / "fish" / "LIAISON.md"
    log_path: Path = HOME / "Docs" / "fish" / "claude_link.jsonl"
    outbox_dir: Path = HOME / "Docs" / "fish" / "say"
    session_name: str = "Fish Liaison"
    timeout_s: float = 180.0
    history_turns: int = 6
    outbox_poll_s: float = 1.0
    outbox_max_age_s: float = 600.0          # stale outbox files are skipped
    allowed_tools: str = "ToolSearch ListAgents SendMessage Read Grep Glob WebSearch WebFetch"

    def validate(self) -> bool:
        if self.timeout_s <= 0:
            raise ClaudeLinkError(f"timeout_s must be positive: {self.timeout_s}")
        if self.outbox_poll_s <= 0:
            raise ClaudeLinkError(f"outbox_poll_s must be positive: {self.outbox_poll_s}")
        return True


def _resolve_binary(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for candidate in (HOME / ".local" / "bin" / name, Path(name).expanduser()):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


class ClaudeLinkService(Service):
    def __init__(self, config: ClaudeLinkConfig = ClaudeLinkConfig()) -> None:
        self.config = config
        # Set by the orchestrator: called with (text, source) from a worker thread.
        self.on_message: Optional[Callable[[str, str], None]] = None
        self._binary: Optional[str] = None
        self._initialized = False
        self._stop = threading.Event()
        self._busy = threading.Lock()
        self._log_lock = threading.Lock()
        self._outbox_thread: Optional[threading.Thread] = None
        self._last_error = ""
        self._last_elapsed_s = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.config.validate()
        if self._initialized:
            return
        self._binary = _resolve_binary(self.config.claude_binary)
        if self._binary is None:
            raise ClaudeLinkError(f"claude binary not found: {self.config.claude_binary}")
        for d in (self.config.workdir, self.config.outbox_dir, self.config.log_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._outbox_thread = threading.Thread(
            target=self._outbox_loop, name="claude-link-outbox", daemon=True
        )
        self._outbox_thread.start()
        self._initialized = True

    def shutdown(self) -> None:
        self._stop.set()
        if self._outbox_thread is not None:
            self._outbox_thread.join(timeout=2)
            self._outbox_thread = None
        self._initialized = False

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "service": "ok" if self._initialized else "uninitialized",
            "binary": self._binary or "",
            "busy": self._busy.locked(),
            "last_elapsed_s": round(self._last_elapsed_s, 1),
            "last_error": self._last_error,
        }

    def reset(self) -> bool:
        self.shutdown()
        try:
            self.initialize()
        except ClaudeLinkError:
            return False
        return True

    # ------------------------------------------------------------------
    # Asking Claude
    # ------------------------------------------------------------------
    def is_busy(self) -> bool:
        return self._busy.locked()

    def ask(self, message: str, *, source: str = "voice") -> bool:
        """Start a background Claude run. Returns False if one is already running."""
        message = message.strip()
        if not message:
            raise ClaudeLinkError("empty message")
        if not self._initialized:
            raise ClaudeLinkError("claude link not initialized")
        if not self._busy.acquire(blocking=False):
            return False
        threading.Thread(
            target=self._ask_worker, args=(message, source), name="claude-link-ask", daemon=True
        ).start()
        return True

    def ask_sync(self, message: str, *, source: str = "cli") -> str:
        """Blocking variant (CLI/tests). Returns the reply text."""
        if not self._busy.acquire(timeout=self.config.timeout_s):
            raise ClaudeLinkError("claude link busy")
        try:
            return self._run_claude(message, source)
        finally:
            self._busy.release()

    def _ask_worker(self, message: str, source: str) -> None:
        try:
            reply = self._run_claude(message, source)
        except Exception as exc:  # never let a worker thread die silently
            self._last_error = str(exc)
            reply = "Sorry, I couldn't reach Claude just now."
        finally:
            self._busy.release()
        self._emit(reply, "claude")

    def _run_claude(self, message: str, source: str) -> str:
        cfg = self.config
        prompt_sys = LIAISON_PROMPT.format(
            log=cfg.log_path, docs=cfg.docs_path, history=self._history_text()
        )
        cmd = [
            self._binary or "claude", "-p", message,
            "--append-system-prompt", prompt_sys,
            "--allowedTools", cfg.allowed_tools,
            "--output-format", "json",
            "-n", cfg.session_name,
        ]
        if cfg.model:
            cmd += ["--model", cfg.model]

        # Don't let a parent Claude session's env leak in (nested-session guards).
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDECODE", "CLAUDE_CODE_"))}
        env.setdefault("HOME", str(HOME))

        started = time.monotonic()
        entry: dict = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source, "message": message}
        try:
            proc = subprocess.run(
                cmd, cwd=cfg.workdir, env=env, capture_output=True, text=True,
                timeout=cfg.timeout_s, stdin=subprocess.DEVNULL,
            )
            self._last_elapsed_s = time.monotonic() - started
            entry["elapsed_s"] = round(self._last_elapsed_s, 1)
            data = json.loads(proc.stdout) if proc.stdout.strip() else {}
            reply = str(data.get("result") or "").strip()
            entry["session_id"] = data.get("session_id")
            entry["cost_usd"] = data.get("total_cost_usd")
            if proc.returncode != 0 or data.get("is_error") or not reply:
                err = (proc.stderr or reply or f"exit {proc.returncode}").strip()[-300:]
                self._last_error = err
                entry["error"] = err
                reply = "Sorry, Claude didn't answer that one."
            else:
                self._last_error = ""
        except subprocess.TimeoutExpired:
            self._last_error = f"timed out after {cfg.timeout_s:.0f}s"
            entry["error"] = self._last_error
            reply = "Claude is taking too long. I'll leave it there."
        except (OSError, ValueError) as exc:
            self._last_error = str(exc)
            entry["error"] = self._last_error
            reply = "Sorry, I couldn't reach Claude just now."
        entry["reply"] = reply
        self._log(entry)
        return reply

    # ------------------------------------------------------------------
    # Outbox: Claude -> fish
    # ------------------------------------------------------------------
    def _outbox_loop(self) -> None:
        cfg = self.config
        while not self._stop.wait(cfg.outbox_poll_s):
            try:
                files = sorted(cfg.outbox_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime)
            except OSError:
                continue
            for path in files:
                try:
                    age = time.time() - path.stat().st_mtime
                    text = path.read_text(encoding="utf-8").strip()
                    path.unlink()
                except OSError:
                    continue
                if not text:
                    continue
                entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "outbox", "say": text}
                if age > cfg.outbox_max_age_s:
                    entry["skipped"] = f"stale ({age:.0f}s old)"
                    self._log(entry)
                    continue
                self._log(entry)
                self._emit(text[:1000], "outbox")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _emit(self, text: str, source: str) -> None:
        if self.on_message is not None and text:
            try:
                self.on_message(text, source)
            except Exception as exc:
                self._last_error = f"on_message failed: {exc}"

    def _log(self, entry: dict) -> None:
        with self._log_lock:
            try:
                with open(self.config.log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def _history_text(self) -> str:
        n = self.config.history_turns
        if n <= 0 or not self.config.log_path.exists():
            return "(none)"
        try:
            lines = self.config.log_path.read_text(encoding="utf-8").splitlines()[-200:]
        except OSError:
            return "(none)"
        turns: list[str] = []
        for line in lines:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if "message" in e and "reply" in e:
                turns.append(f'[{e.get("ts","")}] Person: {e["message"]}\n  You: {e["reply"]}')
            elif "say" in e and "skipped" not in e:
                turns.append(f'[{e.get("ts","")}] (Fish announced: {e["say"]})')
        return "\n".join(turns[-n:]) or "(none)"


if __name__ == "__main__":
    import sys

    svc = ClaudeLinkService()
    svc.initialize()
    question = " ".join(sys.argv[1:]) or "Say hello to the fish in one sentence."
    print(svc.ask_sync(question))
    svc.shutdown()
