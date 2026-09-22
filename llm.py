"""
llm.py - Wraps the OpenAI API call for LCD.

Sends the user's message plus any RAG context (contacts/notes) to OpenAI,
and asks it to reply ONLY with a JSON object containing:
  - reply: text to speak back (in Telugu, unless the user spoke English)
  - action: a structured command the Android app should execute

Requires an environment variable OPENAI_API_KEY.
"""

import os
import json
import logging
import re
import time
import requests

log = logging.getLogger("lcd.llm")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

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


def _fallback(reply: str) -> dict:
    """Same shape ask_lcd always returns, used for graceful error replies."""
    return {"reply": reply, "action": {"type": "none", "target": "", "message": ""}}


def ask_lcd(user_message: str, context: dict) -> dict:
    """
    Calls OpenAI and returns a parsed dict: {"reply": str, "action": {...}}
    Retries a couple of times on a transient 429/503 before giving up.
    Network/timeout failures return a friendly fallback dict instead of raising,
    so a slow or unreachable OpenAI call never turns into a 500 in app.py.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set."
        )

    context_block = _build_context_block(context)

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context_block}\n\nUSER MESSAGE:\n{user_message}"},
        ],
        "temperature": 0.4,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }

    resp = None
    try:
        for attempt in range(3):
            resp = requests.post(
                OPENAI_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=45,
            )
            if resp.status_code in (429, 503) and attempt < 2:
                log.warning("OpenAI returned %d, retrying (attempt %d)...", resp.status_code, attempt + 1)
                time.sleep(2)
                continue
            break

        resp.raise_for_status()

    except requests.exceptions.ReadTimeout:
        log.warning("OpenAI request timed out after 45s")
        return _fallback("Sorry, that took too long to respond. Try again?")

    except requests.exceptions.ConnectionError:
        log.warning("Could not connect to OpenAI")
        return _fallback("Couldn't reach the AI service right now. Try again in a bit.")

    except requests.exceptions.HTTPError:
        log.warning("OpenAI returned HTTP error %s", resp.status_code if resp is not None else "?")
        return _fallback("AI service returned an error. Try again?")

    except requests.exceptions.RequestException as e:
        log.warning("OpenAI request failed: %s", e)
        return _fallback("Something went wrong talking to OpenAI. Try again?")

    data = resp.json()

    raw_text = data["choices"][0]["message"]["content"]
    cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("Failed to parse OpenAI JSON output, raw text was: %s", raw_text)
        # Response was likely cut off mid-JSON (hit max_tokens). Try to
        # salvage just the "reply" field's text so we still speak something
        # sensible instead of the broken JSON string itself.
        match = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)', cleaned)
        if match:
            salvaged = match.group(1).encode().decode("unicode_escape", errors="ignore")
            parsed = {"reply": salvaged, "action": {"type": "none"}}
        else:
            parsed = {
                "reply": "క్షమించండి, నాకు సరిగ్గా అర్థం కాలేదు. మళ్ళీ చెప్పగలరా?",
                "action": {"type": "none"},
            }

    parsed.setdefault("reply", "")
    action = parsed.setdefault("action", {})
    action.setdefault("type", "none")
    action.setdefault("target", "")
    action.setdefault("message", "")

    return parsed
