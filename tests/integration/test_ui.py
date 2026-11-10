from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
import requests


class FakeResponse:
    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


def run_app(monkeypatch, response, post_response=None):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    api_url = f"https://api-{uuid4().hex}.example.test"
    requested_urls = []
    monkeypatch.setenv("API_URL", api_url)

    def fake_get(url, timeout):
        requested_urls.append(url)
        return response

    monkeypatch.setattr("requests.get", fake_get)
    if post_response is not None:
        monkeypatch.setattr("requests.post", lambda *args, **kwargs: post_response)
    path = Path(__file__).resolve().parents[2] / "ui" / "app.py"
    return AppTest.from_file(str(path)).run(timeout=10), api_url, requested_urls


def test_ui_uses_configured_api_and_renders_chat(monkeypatch):
    health = FakeResponse({"status": "ok", "qdrant_docs": 77})
    answer = FakeResponse(
        {
            "answer": "Fake answer [chunk_1].",
            "citations": ["chunk_1"],
            "steps": ["plan", "retrieve", "generate"],
            "status": "answered",
            "stop_reason": "completed",
            "system": "B4",
            "latency_s": 0.1,
        }
    )
    app, api_url, requested_urls = run_app(monkeypatch, health, answer)

    assert not app.exception
    assert app.success
    assert requested_urls == [f"{api_url}/health"]
    app.chat_input[0].set_value("What is attention?").run()
    assert not app.exception
    assert any("Fake answer" in item.value for item in app.markdown)


def test_ui_shows_cold_start_on_timeout(monkeypatch):
    def timeout(url, timeout):
        raise requests.Timeout("API is waking")

    monkeypatch.setenv("API_URL", f"https://api-{uuid4().hex}.example.test")
    monkeypatch.setattr("requests.get", timeout)
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    path = Path(__file__).resolve().parents[2] / "ui" / "app.py"
    app = AppTest.from_file(str(path)).run(timeout=10)

    assert not app.exception
    assert app.warning
    assert app.info
    assert app.button[0].label == "Retry health check"


def test_ui_treats_qdrant_degraded_as_unreachable(monkeypatch):
    degraded = FakeResponse({"status": "degraded", "qdrant_docs": -1}, status_code=503)
    app, _, _ = run_app(monkeypatch, degraded)

    assert not app.exception
    assert app.error


def test_ui_offers_all_three_baselines(monkeypatch):
    health = FakeResponse({"status": "ok", "qdrant_docs": 77})
    at, _, _ = run_app(monkeypatch, health)

    assert not at.exception
    assert list(at.sidebar.radio[0].options) == ["B4", "B1", "B0"]


def test_ui_sends_a_stable_session_id(monkeypatch):
    health = FakeResponse({"status": "ok", "qdrant_docs": 77})
    answer = FakeResponse(
        {
            "answer": "Attention maps queries to keys [c1].",
            "citations": ["c1"],
            "steps": ["plan", "finalize"],
            "status": "answered",
            "stop_reason": "grounded",
            "system": "B4",
            "latency_s": 1.0,
            "standalone_question": "What is self-attention?",
        }
    )
    posted = []

    at, _, _ = run_app(monkeypatch, health, post_response=answer)
    monkeypatch.setattr(
        "requests.post", lambda *args, **kwargs: posted.append(kwargs.get("json")) or answer
    )
    at.chat_input[0].set_value("What is self-attention?").run()

    assert not at.exception
    assert posted, "the UI did not call /query"
    assert posted[0]["session_id"]
    assert posted[0]["system"] == "B4"
