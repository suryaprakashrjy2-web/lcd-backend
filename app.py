"""
LCD Backend Server
-------------------
This is the "brain" server for the LCD personal voice assistant.

Flow:
  Phone (Android app) does Speech-to-Text on-device (Telugu) ->
  sends plain text to this server's /chat endpoint ->
  server looks up relevant info (contacts, notes) via RAG ->
  server calls Gemini LLM to generate a reply + decide an action ->
  server returns JSON { reply, action } ->
  Phone speaks "reply" using on-device Text-to-Speech (Telugu) and executes "action".

Run locally for testing:
    python app.py

Deploy for free (always-on, no laptop needed) on Render.com / Railway.app
using the included requirements.txt (see README.md for exact steps).
"""

import os
import logging
from flask import Flask, request, jsonify

from rag import RAGStore
from llm import ask_lcd

# ---------------------------------------------------------------------------
# Logging setup - keep this format, it's used for debugging on the free host's
# log viewer (Render/Railway both show stdout logs in real time).
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("lcd.app")

app = Flask(__name__)

# One shared RAG store instance (loads/persists to local ./chroma_data folder)
rag_store = RAGStore()


@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint - Render/Railway will ping this."""
    return jsonify({"status": "ok"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main endpoint called by the Android app.

    Request JSON:
        { "message": "<recognized Telugu/English text from the phone>" }

    Response JSON:
        {
          "reply": "<text for the phone to speak back, in Telugu>",
          "action": {
              "type": "open_whatsapp" | "open_gallery" | "call" | "send_message" | "none",
              "target": "<contact name or number, if relevant>",
              "message": "<message body, only for send_message>"
          }
        }
    """
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Missing 'message' field"}), 400

    log.info("Incoming message: %s", user_message)

    # 1. Pull any relevant context (contacts, saved notes) from RAG store
    context = rag_store.query(user_message)

    # 2. Ask Gemini for a reply + structured action
    try:
        result = ask_lcd(user_message, context)
    except Exception as exc:  # noqa: BLE001 - want to always return JSON to the app
        log.exception("LLM call failed")
        return jsonify({
            "reply": "క్షమించండి, ఏదో సమస్య వచ్చింది.",  # "Sorry, something went wrong."
            "action": {"type": "none"},
            "error": str(exc),
        }), 500

    log.info("Reply: %s | Action: %s", result.get("reply"), result.get("action"))
    return jsonify(result), 200


@app.route("/contacts", methods=["POST"])
def add_contact():
    """
    Add or update a contact in the RAG store so LCD can resolve names to
    numbers when you say things like "Call Amma" or "Message Ravi".

    Request JSON: { "name": "Ravi", "phone": "+91XXXXXXXXXX", "notes": "college friend" }
    """
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip()
    notes = (data.get("notes") or "").strip()

    if not name or not phone:
        return jsonify({"error": "Both 'name' and 'phone' are required"}), 400

    rag_store.add_contact(name, phone, notes)
    return jsonify({"status": "saved", "name": name}), 200


@app.route("/notes", methods=["POST"])
def add_note():
    """
    Add a general free-text note/fact to the RAG store so LCD can recall it
    later (e.g. "My wifi password is ..." or "My doctor's number is ...").

    Request JSON: { "text": "..." }
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "'text' is required"}), 400

    rag_store.add_note(text)
    return jsonify({"status": "saved"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # host=0.0.0.0 so it's reachable from Render/Railway's network, and from
    # your phone if you ever run this locally on the same Wi-Fi for testing.
    app.run(host="0.0.0.0", port=port, debug=False)
