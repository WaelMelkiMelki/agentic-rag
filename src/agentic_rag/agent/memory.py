"""Per-session conversation memory.

The API is stateless by default; callers that pass a ``session_id`` get multi-turn
behaviour. History is stored as a LangChain ``BaseChatMessageHistory`` inside a
bounded LRU map, so a long-running process cannot grow without limit.
(The bundled ``InMemoryChatMessageHistory`` is deprecated as of LangChain 1.6, so
this module ships its own list-backed implementation of the same interface.)

Follow-up questions ("and what about the second one?") are turned into a standalone
question before retrieval, using a LangChain prompt template + the same fallback
router as everything else. The shared ``GENERATE_PROMPT`` is untouched, so the
B1-vs-B4 comparison stays valid.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from agentic_rag.llm.chains import text_chain

CONDENSE_PROMPT = """Rewrite the follow-up question as a standalone question that makes sense on its own.
Keep the user's intent and wording where possible. Do not answer the question.
Respond with ONLY the rewritten question.

Conversation so far:
{history}

Follow-up question: {question}

Standalone question:"""

CONDENSE_CHAT_PROMPT = ChatPromptTemplate.from_template(CONDENSE_PROMPT)

DEFAULT_MAX_SESSIONS = 500
DEFAULT_MAX_TURNS = 6


class ListChatMessageHistory(BaseChatMessageHistory):
    """Minimal in-memory history (the stdlib implementation is deprecated)."""

    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def clear(self) -> None:
        self._messages.clear()


class SessionMemory:
    """Bounded, thread-safe store of per-session chat histories."""

    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS, max_turns: int = DEFAULT_MAX_TURNS):
        if max_sessions <= 0:
            raise ValueError("max_sessions must be greater than zero")
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self._sessions: OrderedDict[str, BaseChatMessageHistory] = OrderedDict()
        self._lock = threading.Lock()

    def history(self, session_id: str) -> BaseChatMessageHistory:
        """Return (creating if needed) the history for a session."""
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                self._sessions.move_to_end(session_id)
                return existing
            created = ListChatMessageHistory()
            self._sessions[session_id] = created
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return created

    def remember(self, session_id: str, question: str, answer: str) -> None:
        """Append one user/assistant turn and trim the session to ``max_turns``."""
        history = self.history(session_id)
        with self._lock:
            history.add_message(HumanMessage(content=question))
            history.add_message(AIMessage(content=answer))
            messages = history.messages
            if len(messages) > self.max_turns * 2:
                history.clear()
                for message in messages[-self.max_turns * 2:]:
                    history.add_message(message)

    def transcript(self, session_id: str, max_turns: int = 3) -> str:
        """Render the last ``max_turns`` turns as plain text for prompting."""
        with self._lock:
            stored = self._sessions.get(session_id)
            if stored is None:
                return ""
            messages = list(stored.messages)
        if not messages:
            return ""

        lines = []
        for message in messages[-max_turns * 2:]:
            role = "User" if isinstance(message, HumanMessage) else "Assistant"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)

    def clear(self, session_id: str | None = None) -> None:
        """Forget one session, or every session when ``session_id`` is ``None``."""
        with self._lock:
            if session_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)


def standalone_question(question: str, transcript: str, router) -> str:
    """Rewrite a follow-up using the conversation, falling back to the original."""
    if not transcript.strip():
        return question

    try:
        rewritten = text_chain(CONDENSE_CHAT_PROMPT, router).invoke(
            {"history": transcript, "question": question}
        ).strip()
    except Exception:
        return question

    rewritten = rewritten.strip().strip('"').strip()
    if not 3 <= len(rewritten) <= 400:
        return question
    if rewritten.casefold() == question.casefold():
        return question
    return rewritten
