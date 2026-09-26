"""
memory.py — Fishseus's two kinds of memory.

MemoryStore (data/assistant_memory.json) is long-term memory: what the fish
was explicitly asked to keep — the human's name, response preferences, facts,
and free-form notes the owner writes in the web UI. All of it is shown to the
model every turn, so nothing silently falls out of view.

ConversationStore (data/conversation_log.jsonl) is every turn of conversation,
appended as it happens. The turns from the last `recall_hours` (after the most
recent "start fresh") are replayed into the prompt, so the fish keeps the thread
across restarts. The file itself is never trimmed; it doubles as the log the
web UI shows.

Both stores are thread-safe: the voice loop and the web UI touch them
concurrently.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional


MEMORY_VERSION = 2

# Names that mean "we don't actually know the human's name yet".
_PLACEHOLDER_NAMES = {"", "user", "the user", "human", "the human", "none"}

# v1 shipped these preference values as defaults. They duplicated the
# personality prompt, so migration drops them unless the user changed them.
_V1_DEFAULT_PREFERENCES = {
    ("response_style", "brief, helpful, technical when needed"),
    ("humor_level", "medium"),
    ("personality", "dramatic sarcastic fish oracle"),
}

# Words ignored when matching a spoken "forget ..." request against facts.
_FORGET_STOPWORDS = {
    "the", "a", "an", "my", "your", "that", "this", "about", "of", "for",
    "to", "is", "and", "please", "fishseus", "forget", "delete", "remove",
    "erase", "remember", "memory", "everything", "fact",
}


def normalize_key(key: str) -> str:
    """'facts.Ethan Address' -> 'ethan_address' (stable snake_case fact label)."""
    key = str(key or "").strip().lower()
    if key.startswith("facts."):
        key = key[len("facts."):]
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key[:48].rstrip("_")


def is_placeholder_name(name: Any) -> bool:
    return str(name or "").strip().lower() in _PLACEHOLDER_NAMES


def _now() -> int:
    return int(time.time())


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write via a temp file + rename so a power cut never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------
# Long-term memory
# ----------------------------------------------------------------------

class MemoryStore:
    """
    JSON-backed long-term memory.

    Layout (version 2):
        {
          "version": 2,
          "profile":     {"assistant_name": "Fishseus", "user_name": "Caleb"},
          "preferences": {"response_style": "short answers"},
          "notes":       "free text from the owner, always in context",
          "facts": [{"key": "ethan_address",
                     "value": "Ethan Weber lives at 5450 Lockwood Rd, Madison, Ohio.",
                     "created_at": 1782681559, "updated_at": 1782681559}]
        }

    Facts are keyed by a stable snake_case label and upserted: writing an
    existing label replaces its sentence instead of piling up a near-duplicate.
    """

    def __init__(self, path: Path, assistant_name: str, user_name: str) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.data: dict[str, Any] = self._empty(assistant_name, user_name)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self.save()
                return
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("top level is not an object")
            except Exception as exc:
                # Keep the unreadable file for inspection rather than letting the
                # next save() overwrite whatever it held.
                backup = self.path.with_name(f"{self.path.stem}.corrupt-{_now()}{self.path.suffix}")
                try:
                    self.path.replace(backup)
                except OSError:
                    pass
                print(f"[memory] Could not read {self.path.name} ({exc}); "
                      f"moved it to {backup.name} and started fresh", flush=True)
                self.save()
                return

            profile = self.data["profile"]
            migrated = self._migrate(loaded, profile["assistant_name"], profile["user_name"])
            self.data = migrated
            if loaded.get("version") != MEMORY_VERSION:
                print(f"[memory] Migrated {self.path.name} to version {MEMORY_VERSION}", flush=True)
                self.save()

    def save(self) -> None:
        with self._lock:
            _write_json_atomic(self.path, self.data)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.data)

    # ------------------------------------------------------------------
    # Profile / preferences / notes
    # ------------------------------------------------------------------
    @property
    def user_name(self) -> str:
        """The human's name, or "" while it's unknown."""
        name = str(self.data.get("profile", {}).get("user_name") or "").strip()
        return "" if is_placeholder_name(name) else name

    def set_profile(self, field: str, value: Any) -> None:
        with self._lock:
            self.data["profile"][str(field)] = str(value or "").strip()

    def set_preference(self, key: str, value: Any) -> None:
        """Set a preference; an empty value removes it."""
        key = normalize_key(key)
        if not key:
            return
        with self._lock:
            text = str(value or "").strip()
            if text:
                self.data["preferences"][key] = text
            else:
                self.data["preferences"].pop(key, None)

    def set_notes(self, text: str) -> None:
        with self._lock:
            self.data["notes"] = str(text or "").strip()

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------
    def facts(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self.data["facts"])

    def has_fact(self, key: str) -> bool:
        key = normalize_key(key)
        with self._lock:
            return any(f["key"] == key for f in self.data["facts"])

    def remember(self, key: str, value: Any) -> tuple[str, str]:
        """
        Upsert a fact. Returns (status, key) where status is "added", "updated",
        "unchanged" or "ignored" (empty key/value).
        """
        value = " ".join(str(value or "").split())
        key = normalize_key(key) or normalize_key(" ".join(value.split()[:5]))
        if not key or not value:
            return "ignored", key
        now = _now()
        with self._lock:
            facts = self.data["facts"]
            for fact in facts:
                if fact["key"] == key:
                    if fact["value"] == value:
                        return "unchanged", key
                    fact["value"] = value
                    fact["updated_at"] = now
                    return "updated", key
            # The same sentence under a different label is still one fact.
            for fact in facts:
                if fact["value"].lower() == value.lower():
                    return "unchanged", fact["key"]
            facts.append({"key": key, "value": value, "created_at": now, "updated_at": now})
            return "added", key

    def rename_fact(self, old_key: str, new_key: str) -> bool:
        old_key, new_key = normalize_key(old_key), normalize_key(new_key)
        if not old_key or not new_key:
            return False
        with self._lock:
            if old_key != new_key and self.has_fact(new_key):
                return False
            for fact in self.data["facts"]:
                if fact["key"] == old_key:
                    fact["key"] = new_key
                    fact["updated_at"] = _now()
                    return True
        return False

    def delete_fact(self, key: str) -> bool:
        key = normalize_key(key)
        with self._lock:
            before = len(self.data["facts"])
            self.data["facts"] = [f for f in self.data["facts"] if f["key"] != key]
            return len(self.data["facts"]) != before

    def clear_facts(self) -> None:
        with self._lock:
            self.data["facts"] = []

    def forget(self, query: str) -> list[dict[str, Any]]:
        """
        Remove facts matching a spoken description and return what was removed.

        An exact label match wins. Otherwise every meaningful word in the query
        must appear in a fact's label or sentence. Matches nothing (and changes
        nothing) when unsure. Callers persist via save().
        """
        q = str(query or "").strip().lower()
        if q.startswith("facts."):
            q = q[len("facts."):]
        if not q:
            return []
        with self._lock:
            facts = self.data["facts"]
            exact = [f for f in facts if f["key"] == normalize_key(q)]
            if exact:
                matches = exact
            else:
                tokens = [t for t in re.split(r"[^a-z0-9]+", q)
                          if len(t) >= 2 and t not in _FORGET_STOPWORDS]
                if not tokens:
                    return []

                def haystack(f: dict[str, Any]) -> str:
                    return f"{f['key'].replace('_', ' ')} {f['value']}".lower()

                matches = [f for f in facts if all(t in haystack(f) for t in tokens)]
            if not matches:
                return []
            doomed = {f["key"] for f in matches}
            self.data["facts"] = [f for f in facts if f["key"] not in doomed]
            return copy.deepcopy(matches)

    # ------------------------------------------------------------------
    # Model-driven updates
    # ------------------------------------------------------------------
    def apply_update(self, key: str, value: Any) -> Optional[dict[str, Any]]:
        """
        Apply one {"key", "value"} update proposed by the model.

        "profile.X" and "preferences.X" set those fields; "facts.X" (or a bare
        label) upserts a fact. Anything else is ignored. Returns the update as
        stored, or None when nothing changed.
        """
        key = str(key or "").strip()
        if not key or value is None:
            return None
        lowered = key.lower()
        if lowered.startswith("profile."):
            field = normalize_key(key[len("profile."):])
            if not field or not str(value).strip():
                return None
            self.set_profile(field, value)
            return {"key": f"profile.{field}", "value": str(value).strip()}
        if lowered.startswith("preferences."):
            pref = normalize_key(key[len("preferences."):])
            if not pref:
                return None
            self.set_preference(pref, value)
            return {"key": f"preferences.{pref}", "value": str(value).strip()}
        if lowered.startswith(("session.", "notes", "version")):
            return None  # v1 scratch state / owner-only fields
        status, fact_key = self.remember(key, value)
        if status in ("added", "updated"):
            return {"key": f"facts.{fact_key}", "value": " ".join(str(value).split()), "status": status}
        return None

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------
    def prompt_block(self) -> str:
        """Long-term memory, framed for the model."""
        with self._lock:
            name = self.user_name
            who = name or "the human"
            prefs = dict(self.data["preferences"])
            notes = self.data["notes"]
            facts = list(self.data["facts"])

        lines = ["LONG-TERM MEMORY"]
        if name:
            lines.append(f"The human you usually talk with is {name}.")
        else:
            lines.append("You don't know the human's name yet.")

        if prefs:
            lines.append("")
            lines.append(f"How {who} wants you to respond:")
            lines += [f"- {k.replace('_', ' ')}: {v}" for k, v in prefs.items()]

        if notes:
            lines.append("")
            lines.append(f"Standing notes from {who} (always true unless they say otherwise):")
            lines.append(notes)

        lines.append("")
        if facts:
            lines.append("Facts you were asked to remember. The [labels] are internal ids for "
                         "updating or forgetting a fact; never say them aloud.")
            lines += [f"- [{f['key']}] {f['value']}" for f in facts]
        else:
            lines.append("You haven't been asked to remember any facts yet.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    @staticmethod
    def _empty(assistant_name: str, user_name: str) -> dict[str, Any]:
        return {
            "version": MEMORY_VERSION,
            "profile": {
                "assistant_name": assistant_name,
                "user_name": "" if is_placeholder_name(user_name) else user_name,
            },
            "preferences": {},
            "notes": "",
            "facts": [],
        }

    @classmethod
    def _migrate(cls, loaded: dict[str, Any], assistant_name: str, user_name: str) -> dict[str, Any]:
        """Normalise any v1/v2 file into the v2 layout (tolerant of hand edits)."""
        data = cls._empty(assistant_name, user_name)

        profile = loaded.get("profile")
        if isinstance(profile, dict):
            for key, value in profile.items():
                if isinstance(value, (str, int, float)) and str(value).strip():
                    data["profile"][str(key)] = str(value).strip()
        if is_placeholder_name(data["profile"].get("user_name")):
            data["profile"]["user_name"] = ""

        prefs = loaded.get("preferences")
        if isinstance(prefs, dict):
            for key, value in prefs.items():
                if value is None or not str(value).strip():
                    continue
                if (key, value) in _V1_DEFAULT_PREFERENCES and loaded.get("version") != MEMORY_VERSION:
                    continue
                data["preferences"][normalize_key(key) or str(key)] = str(value).strip()

        if isinstance(loaded.get("notes"), str):
            data["notes"] = loaded["notes"].strip()

        facts = loaded.get("facts")
        if isinstance(facts, list):
            seen: dict[str, dict[str, Any]] = {}
            for fact in facts:
                if not isinstance(fact, dict):
                    continue
                value = " ".join(str(fact.get("value") or "").split())
                key = normalize_key(fact.get("key") or "") or normalize_key(" ".join(value.split()[:5]))
                if not key or not value:
                    continue
                created = int(fact.get("created_at") or _now())
                entry = {
                    "key": key,
                    "value": value,
                    "created_at": created,
                    "updated_at": int(fact.get("updated_at") or created),
                }
                # v1 appended duplicates; the newest value for a label wins.
                if key not in seen or entry["updated_at"] >= seen[key]["updated_at"]:
                    seen[key] = entry
            data["facts"] = sorted(seen.values(), key=lambda f: f["created_at"])

        return data


# ----------------------------------------------------------------------
# Conversation memory
# ----------------------------------------------------------------------

class ConversationStore:
    """
    Append-only JSONL record of every conversational turn.

    Each line is a turn:
        {"id", "timestamp", "source": "voice"|"web"|"sensor", "user",
         "speak", "motion", "tool_calls", "tool_results", "memory_updates",
         "answer", "answer_motion", "assistant", "elapsed_s"}
    where "speak" is the model's first reply, "answer" is what it said after
    tool results came back, and "assistant" is everything actually spoken.
    Older (pre-v2) lines only have user/assistant/tool_calls/tool_results and
    still load.

    "Start fresh" appends {"event": "clear"}; turns before it are never
    recalled again but stay in the log.
    """

    CLEAR_EVENT = "clear"

    def __init__(self, path: Path, recall_hours: float = 12.0, max_recall_turns: int = 150) -> None:
        self.path = Path(path)
        self.recall_hours = float(recall_hours)
        self.max_recall_turns = int(max_recall_turns)
        self._lock = threading.RLock()
        self._turns: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Loading / recall
    # ------------------------------------------------------------------
    def load(self) -> None:
        """(Re)load the recall window from disk."""
        with self._lock:
            self._turns = []
            for record in self._read_all():
                if record.get("event") == self.CLEAR_EVENT:
                    self._turns = []
                elif "user" in record:
                    self._turns.append(record)
            self._prune()

    def configure(self, recall_hours: float, max_recall_turns: int) -> None:
        with self._lock:
            self.recall_hours = float(recall_hours)
            self.max_recall_turns = int(max_recall_turns)
            self.load()

    def recent(self) -> list[dict[str, Any]]:
        """Turns inside the recall window, oldest first."""
        with self._lock:
            self._prune()
            return copy.deepcopy(self._turns)

    def _prune(self) -> None:
        if self.recall_hours > 0:
            cutoff = time.time() - self.recall_hours * 3600
            self._turns = [t for t in self._turns if float(t.get("timestamp") or 0) >= cutoff]
        if self.max_recall_turns > 0 and len(self._turns) > self.max_recall_turns:
            self._turns = self._turns[-self.max_recall_turns:]

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def append(self, turn: dict[str, Any]) -> dict[str, Any]:
        turn = dict(turn)
        turn.setdefault("id", uuid.uuid4().hex[:12])
        turn.setdefault("timestamp", _now())
        with self._lock:
            self._append_line(turn)
            self._turns.append(turn)
            self._prune()
        return turn

    def clear(self) -> int:
        """Start fresh: stop recalling everything said so far. Returns turns dropped."""
        with self._lock:
            dropped = len(self._turns)
            self._append_line({"id": uuid.uuid4().hex[:12], "timestamp": _now(), "event": self.CLEAR_EVENT})
            self._turns = []
            return dropped

    def delete(self, turn_id: str) -> bool:
        """Remove one turn from the log and from recall."""
        if not turn_id:
            return False
        with self._lock:
            records = self._read_all()
            kept = [r for r in records if r.get("id") != turn_id]
            if len(kept) == len(records):
                return False
            self._rewrite(kept)
            self._turns = [t for t in self._turns if t.get("id") != turn_id]
            return True

    def wipe(self) -> None:
        """Delete the whole log (and with it, all recall)."""
        with self._lock:
            self._rewrite([])
            self._turns = []

    # ------------------------------------------------------------------
    # Log access (web UI)
    # ------------------------------------------------------------------
    def log(self, n: int = 30) -> list[dict[str, Any]]:
        """The last n records (turns and clear markers), oldest first."""
        with self._lock:
            records = self._read_all()
        return records[-n:] if n > 0 else records

    def recalled_ids(self) -> set[str]:
        with self._lock:
            self._prune()
            return {t.get("id") for t in self._turns if t.get("id")}

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except OSError as exc:
            print(f"[conversation] Could not read {self.path.name}: {exc}", flush=True)
        return records

    def _append_line(self, record: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            print(f"[conversation] Could not write {self.path.name}: {exc}", flush=True)

    def _rewrite(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        tmp.replace(self.path)
