"""Langfuse tracing for agent invocations.

Tracing is **optional and fail-safe**: without ``LANGFUSE_PUBLIC_KEY`` /
``LANGFUSE_SECRET_KEY`` (or without the SDK installed) every function here is a
no-op that returns ``""``. A tracing failure must never break a query, so all
SDK interaction is wrapped in ``try``/``except``.

The Langfuse Python SDK changed its public API three times:

* v2 — ``langfuse.trace(...)``
* v3 — ``langfuse.start_as_current_span(...)``
* v4/v5 — ``langfuse.start_as_current_observation(...)``

``trace_agent`` detects which one the installed SDK provides, so the project
works with whatever ``uv sync`` resolves instead of pinning a fragile version.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

TRACE_NAME = "agentic-rag-query"

# Sentinel: build the client lazily on first use, never at import time, so that
# importing this module (tests, CLI runs, Render cold start) cannot fail.
_UNINITIALIZED = object()
_client: Any = _UNINITIALIZED


def _langfuse_configured() -> bool:
    """True when both Langfuse credentials are present in the environment."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(
        os.getenv("LANGFUSE_SECRET_KEY")
    )


def get_client() -> Any:
    """Return a Langfuse client, or ``None`` when tracing is not configured."""
    global _client
    if _client is not _UNINITIALIZED:
        return _client

    _client = None
    if not _langfuse_configured():
        logger.info("Langfuse keys not set; tracing disabled")
        return _client

    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("langfuse is not installed; tracing disabled")
        return _client

    kwargs: dict[str, Any] = {
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
    }
    host = os.getenv("LANGFUSE_HOST")
    if host:
        kwargs["host"] = host

    try:
        _client = Langfuse(**kwargs)
    except Exception:  # pragma: no cover - defensive
        logger.warning("Langfuse client could not be created", exc_info=True)
        _client = None
    return _client


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize B1 (``answer``) and B4 (``final_answer``) results."""
    return {
        "answer": result.get("final_answer") or result.get("answer") or "",
        "status": result.get("status", ""),
        "stop_reason": result.get("stop_reason", ""),
        "citations": list(result.get("citations") or []),
        "steps": list(result.get("steps") or []),
        "system": result.get("system", "B4"),
    }


def _emit(
    client: Any,
    question: str,
    payload: dict[str, Any],
    input_value: Any,
) -> str:
    """Send one trace using whichever SDK API surface is available."""
    output = {
        "answer": payload["answer"],
        "status": payload["status"],
        "stop_reason": payload["stop_reason"],
    }
    metadata = {
        "steps": payload["steps"],
        "citations": payload["citations"],
        "system": payload["system"],
    }
    level = "DEFAULT"
    status_message = payload["stop_reason"] or "completed"

    common: dict[str, Any] = {
        "name": TRACE_NAME,
        "input": input_value,
        "output": output,
        "metadata": metadata,
        "level": level,
    }

    # v4 / v5
    if hasattr(client, "start_as_current_observation"):
        with client.start_as_current_observation(
            as_type="span", status_message=status_message, **common
        ):
            return client.get_current_trace_id() or ""

    # v3
    if hasattr(client, "start_as_current_span"):
        with client.start_as_current_span(status_message=status_message, **common):
            return client.get_current_trace_id() or ""

    # v2
    if hasattr(client, "trace"):
        trace = client.trace(input=input_value, output=output, metadata=metadata, name=TRACE_NAME)
        trace.update(level=level, status_message=status_message)
        return trace.id

    logger.warning("Unsupported Langfuse SDK surface; tracing skipped")
    return ""


def trace_agent(
    question: str,
    result: dict[str, Any],
    client: Any = None,
) -> str:
    """Log one agent invocation to Langfuse and return the trace ID.

    Returns ``""`` when tracing is disabled or when the SDK call fails; it never
    raises, so the ``/query`` endpoint cannot break because of observability.
    """
    resolved = get_client() if client is None else client
    if resolved is None:
        return ""

    payload = _payload(result)
    try:
        return _emit(resolved, question, payload, {"question": question})
    except Exception:  # pragma: no cover - defensive
        logger.warning("Langfuse trace failed", exc_info=True)
        return ""


def flush(client: Any = None) -> None:
    """Flush buffered traces (call on shutdown or after batch runs)."""
    resolved = get_client() if client is None else client
    if resolved is None:
        return
    try:
        resolved.flush()
    except Exception:  # pragma: no cover - defensive
        logger.warning("Langfuse flush failed", exc_info=True)


def reset_client() -> None:
    """Drop the cached client (used by tests and after env changes)."""
    global _client
    _client = _UNINITIALIZED
