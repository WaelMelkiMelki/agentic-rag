"""Streamlit chat UI; all agent work is delegated to the FastAPI service."""
from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic RAG", page_icon="🔍", layout="wide")
st.title("🔍 Agentic RAG")
st.caption("Document question answering · B1 = plain RAG · B4 = full agent")


@st.cache_data(ttl=30)
def check_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"status": "unreachable", "qdrant_docs": -1}


health = check_health()
if health["status"] == "ok":
    st.sidebar.success(f"API online · {health['qdrant_docs']} chunks in Qdrant")
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
