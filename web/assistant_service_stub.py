"""
assistant_service_stub.py
=======================

This is a very small stand‑in for the full assistant logic used in the
Fishseus project.  The real project uses a sophisticated prompt
builder, memory store and LLM interface to generate JSON responses.
Since those components are not available in this environment, this
stub provides minimal functionality to allow the web UI to display
responses when a user submits a message.

The core function exported by this module is `handle_user_text`.  It
takes a plain string and returns a dictionary with a single key,
``"reply"``, containing text that the UI can display.  You can
customise this stub to return different responses for certain
questions, or to simulate the Fishseus personality.

Usage:

    from assistant_service_stub import handle_user_text
    response = handle_user_text("Who are you?")
    print(response["reply"])

In a production system, this function would pass the text through
speech‑to‑text, send it to the assistant service (which would in turn
call an LLM via `llm_service`), parse the returned JSON and handle
tool calls.  Here we simply echo a canned response based on simple
pattern matching.
"""

from __future__ import annotations

import re
from typing import Dict


def handle_user_text(text: str) -> Dict[str, str]:
    """Generate a reply for a given input text.

    This stub uses very naive pattern matching to recognise a few
    special prompts.  In all other cases it will simply echo the
    user's message prefaced with a friendly prefix.

    Args:
        text: A user message in plain text.

    Returns:
        A dict with a single key ``"reply"`` containing the fish's
        response.
    """
    text = text.strip().lower()
    # Respond to introductions
    if not text:
        return {"reply": "I didn't hear anything."}
    if re.search(r"who\s+are\s+you", text):
        return {"reply": "I am Fishseus, the wisest fish on the wall."}
    if re.search(r"what\s+is\s+your\s+name", text):
        return {"reply": "You can call me Fishseus."}
    if re.search(r"hello|hi|hey", text):
        return {"reply": "Hello there! I'm always ready to chat."}
    # Generic response: echo the user's message
    return {"reply": f"You said: {text}"}
