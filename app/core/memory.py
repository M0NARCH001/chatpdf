"""
Session memory using langchain_community.chat_message_histories.

Stores the last MAX_TURNS exchanges per session in memory.
For production, swap ChatMessageHistory for a Redis-backed equivalent.
"""
import logging
from typing import Dict
from langchain_community.chat_message_histories import ChatMessageHistory

logger = logging.getLogger(__name__)

MAX_TURNS = 5   # keep last N human/ai pairs (= 2*N messages)

_SESSION_HISTORIES: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Return (or create) the ChatMessageHistory for a session."""
    if session_id not in _SESSION_HISTORIES:
        _SESSION_HISTORIES[session_id] = ChatMessageHistory()
    history = _SESSION_HISTORIES[session_id]
    # Trim to the most recent MAX_TURNS exchanges
    max_messages = MAX_TURNS * 2
    if len(history.messages) > max_messages:
        history.messages = history.messages[-max_messages:]
    return history


def reset_memory(session_id: str) -> None:
    """Clear the chat history for a session."""
    if session_id in _SESSION_HISTORIES:
        _SESSION_HISTORIES[session_id].clear()
        logger.info("Memory cleared for session '%s'.", session_id)
