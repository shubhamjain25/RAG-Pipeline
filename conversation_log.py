"""
Conversation logging for the chat UI.

Each Streamlit session gets one 16-digit conversation_id. Every Q&A turn
(standard or "think deeper") is appended as a structured log entry that the
eye-icon viewer in app.py can render.
"""

import random
import string
from datetime import datetime, timezone


def generate_conversation_id() -> str:
    """16-digit numeric conversation id, generated once per session."""
    return "".join(random.choices(string.digits, k=16))


def generate_turn_id() -> str:
    """Shorter id for an individual turn within a conversation, used as a stable widget key."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))


def new_log_entry(conversation_id: str, document_id, question: str, metadata: dict, latency_ms: int) -> dict:
    """Build a structured log entry from the metadata dict produced by query_rag_stream."""
    chunks = metadata.get("chunks", []) or []
    return {
        "conversation_id": conversation_id,
        "turn_id": generate_turn_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "document_id": document_id,
        "user_message": question,
        "standalone_question": metadata.get("standalone_question", question),
        "deep_think": bool(metadata.get("deep_think", False)),
        "decomposition_mode": metadata.get("decomposition_mode"),
        "queries_used": metadata.get("queries_used", [question]),
        "retrieved_chunks": [
            {
                "id": c.get("id"),
                "content": c.get("content"),
                "similarity": c.get("similarity"),
            }
            for c in chunks
        ],
        "llm_answer": metadata.get("answer", ""),
        "token_usage": metadata.get("token_usage") or {},
        "latency_ms": latency_ms,
        "error": metadata.get("error"),
    }