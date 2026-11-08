"""Streamlit chat UI; all agent work is delegated to the FastAPI service."""
from __future__ import annotations

import os

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def configured_api_url() -> str:
    """Read the API endpoint from env, Streamlit secrets, or local default."""
    value = os.environ.get("API_URL")
    if not value:
        try:
            value = st.secrets["API_URL"]
        except (KeyError, StreamlitSecretNotFoundError):
            value = "http://localhost:8000"
    return value.rstrip("/")


API_URL = configured_api_url()

st.set_page_config(page_title="Agentic RAG", page_icon="🔍", layout="wide")
st.title("🔍 Agentic RAG")
st.caption("Document question answering · B1 = plain RAG · B4 = full agent")


@st.cache_data(ttl=30)
def check_health(api_url: str):
    try:
        response = requests.get(f"{api_url}/health", timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        return {"status": "cold_start", "qdrant_docs": -1}
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code in {502, 503, 504}:
            try:
                payload = error.response.json()
                if isinstance(payload, dict) and payload.get("status") == "degraded":
                    return {"status": "unreachable", "qdrant_docs": -1}
            except ValueError:
                pass
            return {"status": "cold_start", "qdrant_docs": -1}
        return {"status": "unreachable", "qdrant_docs": -1}
    except requests.RequestException:
        return {"status": "unreachable", "qdrant_docs": -1}


health = check_health(API_URL)
if health["status"] == "ok":
    st.sidebar.success(f"API online · {health['qdrant_docs']} chunks in Qdrant")
elif health["status"] == "cold_start":
    st.sidebar.warning("Render may need about a minute to resume.", title="API is waking up")
    st.info("Wait briefly, then retry the health check.", title="The API is starting")
    if st.button("Retry health check"):
        check_health.clear()
        st.rerun()
    st.stop()
else:
    st.sidebar.error("API unreachable or degraded — check that the API and Qdrant are running.")
    st.stop()

system = st.sidebar.radio("System", ["B4", "B1"], index=0)
question = st.chat_input("Ask a question about the corpus...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "system": system},
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout:
            st.error("The request timed out. The model provider may be slow; try again.")
            st.stop()
        except requests.RequestException as error:
            st.error(f"API request failed: {error}")
            st.stop()

        st.write(data["answer"])
        with st.expander("📎 Citations"):
            if data["citations"]:
                for citation in data["citations"]:
                    st.code(citation)
            else:
                st.write("None")

        with st.expander("🧠 Agent Steps"):
            for index, step in enumerate(data["steps"], 1):
                st.write(f"{index}. {step}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", data["status"])
        col2.metric("Stop reason", data["stop_reason"])
        col3.metric("Latency", f"{data['latency_s']}s")
