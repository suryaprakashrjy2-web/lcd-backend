"""
rag.py - Lightweight, dependency-free "RAG" store for LCD.

Stores two kinds of data in a local JSON file (./lcd_data.json):
  1. Contacts   -> so voice commands like "Call Amma" resolve to a phone number
  2. Notes      -> free-text personal facts you want LCD to remember

Retrieval is simple keyword-overlap matching (no embedding model, no vector
DB, no internet download required). This is intentionally simple: for a
personal assistant used by one person occasionally, a handful of contacts
and notes doesn't need real vector search - keyword matching is fast,
free, and has zero external dependencies or failure points.
"""

import os
import json
import logging

log = logging.getLogger("lcd.rag")

_DATA_FILE = "./lcd_data.json"


def _load() -> dict:
    if os.path.exists(_DATA_FILE):
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"contacts": [], "notes": []}


def _save(data: dict):
    with open(_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _score(query: str, text: str) -> int:
    """Simple keyword-overlap score between the query and a stored text."""
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    return len(q_words & t_words)


class RAGStore:
    def __init__(self):
        self._data = _load()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def add_contact(self, name: str, phone: str, notes: str = ""):
        contacts = self._data["contacts"]
        # Replace existing contact with the same name (case-insensitive), else append.
        for c in contacts:
            if c["name"].lower() == name.lower():
                c.update({"phone": phone, "notes": notes})
                break
        else:
            contacts.append({"name": name, "phone": phone, "notes": notes})
        _save(self._data)
        log.info("Saved contact: %s -> %s", name, phone)

    def add_note(self, text: str):
        self._data["notes"].append(text)
        _save(self._data)
        log.info("Saved note: %s", text)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def query(self, user_message: str, n_results: int = 3) -> dict:
        """
        Returns the most relevant contacts and notes for a given message,
        ranked by simple keyword overlap. Safe to call when data is empty.
        """
        contacts = self._data["contacts"]
        notes = self._data["notes"]

        scored_contacts = sorted(
            contacts,
            key=lambda c: _score(user_message, f"{c['name']} {c.get('notes', '')}"),
            reverse=True,
        )
        top_contacts = [
            {"document": f"{c['name']}: {c['phone']}", "metadata": c}
            for c in scored_contacts[:n_results]
            if _score(user_message, f"{c['name']} {c.get('notes', '')}") > 0
        ]

        scored_notes = sorted(notes, key=lambda n: _score(user_message, n), reverse=True)
        top_notes = [n for n in scored_notes[:n_results] if _score(user_message, n) > 0]

        # If nothing matched by keyword but there are only a few contacts total,
        # include them all anyway so the LLM still has the full picture
        # (helps with commands like "call my mom" where the name itself may
        # not be a literal keyword match).
        if not top_contacts and len(contacts) <= 5:
            top_contacts = [{"document": f"{c['name']}: {c['phone']}", "metadata": c} for c in contacts]

        return {"contacts": top_contacts, "notes": top_notes}
