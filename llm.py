"""
llm.py - Wraps the Gemini API call for LCD.

Sends the user's message plus any RAG context (contacts/notes) to Gemini,
and asks it to reply ONLY with a JSON object containing:
  - reply: text to speak back (in Telugu, unless the user spoke English)
  - action: a structured command the Android app should execute

Requires an environment variable GEMINI_API_KEY (free tier from
https://aistudio.google.com/apikey).
"""

import os
import json
import logging
import requests

log = logging.getLogger("lcd.llm")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

SYSTEM_PROMPT = """You are LCD, a personal voice assistant for a single user, similar to \
Jarvis from Iron Man. The user speaks to you mostly in Telugu, sometimes English.

You must decide two things for every message:
1. What to SAY back (a short, natural, spoken-style reply - in Telugu unless the \
user's message was in English).
2. What ACTION the phone should take, if any.

Valid action types:
- "open_whatsapp"   -> target: contact name if mentioned, else empty
- "open_gallery"    -> target: "" (no target needed)
- "call"            -> target: contact name or phone number to call
- "send_message"    -> target: contact name or phone number, message: text body
- "none"            -> when it's just a question/conversation, no phone action needed

Use the CONTEXT block (contacts/notes retrieved from the user's own saved data) to \
resolve names to phone numbers whenever possible. If a contact name is mentioned but \
not found in context, still set the action target to the spoken name as plain text - \
the app will try to resolve it locally.

Respond with ONLY a raw JSON object, no markdown formatting, no backticks, no \
extra commentary. Exact shape:
{"reply": "...", "action": {"type": "...", "target": "...", "message": "..."}}
"""


def _build_context_block(context: dict) -> str:
    lines = []
    if context.get("contacts"):
        lines.append("Known contacts:")
        for c in context["contacts"]:
            meta = c["metadata"]
            lines.append(f"- {meta.get('name')}: {meta.get('phone')} ({meta.get('notes', '')})")
    if context.get("notes"):
        lines.append("Saved notes:")
        for n in context["notes"]:
            lines.append(f"- {n}")
    return "\n".join(lines) if lines else "No relevant saved data found."


def ask_lcd(user_message: str, context: dict) -> dict:
    """
    Calls Gemini and returns a parsed dict: {"reply": str, "action": {...}}
    Falls back to a safe default if parsing fails.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    context_block = _build_context_block(context)
    full_prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_block}\n\nUSER MESSAGE:\n{user_message}"

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 300,
        },
    }

    resp = requests.post(
        GEMINI_URL,
        json=payload,
        headers={
            "x-goog-api-key": GEMINI_API_KEY,
            "Content-Type": "application/json",
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("Failed to parse Gemini JSON output, raw text was: %s", raw_text)
        parsed = {"reply": raw_text, "action": {"type": "none"}}

    parsed.setdefault("reply", "")
    action = parsed.setdefault("action", {})
    action.setdefault("type", "none")
    action.setdefault("target", "")
    action.setdefault("message", "")

    return parsed
