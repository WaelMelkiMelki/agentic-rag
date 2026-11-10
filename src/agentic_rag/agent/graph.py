import re

from langgraph.graph import END, START, StateGraph

from agentic_rag.agent.grade import grade_chunks
from agentic_rag.agent.ground import check_grounding
from agentic_rag.agent.parsing import parse_plan
from agentic_rag.agent.plan import PLAN_CHAT_PROMPT
from agentic_rag.agent.prompts import ABSTENTION, GENERATE_CHAT_PROMPT, format_context
from agentic_rag.agent.rewrite import rewrite_query
from agentic_rag.agent.state import AgentState
from agentic_rag.llm.chains import text_chain


def build_agent(router, embedder, store, *, top_k: int = 5, max_router_calls: int = 10,
                max_rounds: int = 2, grounded_threshold: float = 0.75):
    """Build a bounded RAG graph with grading, grounding, and query retries.

    ``max_rounds`` counts retries after the initial retrieval round.
    """
    if top_k < 1 or max_router_calls < 1 or max_rounds < 0:
        raise ValueError("top_k and max_router_calls must be positive; max_rounds cannot be negative")

    # LCEL chains: prompt -> fallback router -> text. Built once per agent.
    plan_chain = text_chain(PLAN_CHAT_PROMPT, router)
    generate_chain = text_chain(GENERATE_CHAT_PROMPT, router)

    def initialize(state: AgentState) -> dict:
        """Fill graph bookkeeping defaults so callers only need to pass a question."""
        return {
            "router_calls": state.get("router_calls", 0),
            "steps": state.get("steps", []),
        }

    def plan(state: AgentState) -> dict:
        if state["router_calls"] >= max_router_calls:
            return {"route": "abstain", "stop_reason": "budget_exhausted",
                    "steps": ["Planner skipped: budget exhausted"]}
        try:
            raw = plan_chain.invoke({"question": state["question"]})
        except Exception:
            return {"route": "abstain", "router_calls": state["router_calls"] + 1,
                    "stop_reason": "llm_unavailable", "steps": ["Planner failed: LLM unavailable"]}
        parsed = parse_plan(raw)
        if parsed is not None:
            route, query = parsed.action, parsed.query.strip() or state["question"]
            step = f"Plan: {route}"
        else:
            route, query, step = "retrieve", state["question"], "Plan invalid; searching original question"
        return {"route": route, "search_query": query, "router_calls": state["router_calls"] + 1,
                "round": 0, "steps": [step]}

    def retrieve(state: AgentState) -> dict:
        try:
            vector = embedder.embed_query(state.get("search_query", state["question"]))
            chunks = store.search(vector, top_k=top_k)
        except Exception:
            return {"chunks": [], "relevant_chunks": [], "steps": ["Retrieval failed"]}
        return {"chunks": chunks, "relevant_chunks": [],
                "steps": [f"Round {state.get('round', 0)}: retrieved {len(chunks)} chunks"]}

    def grade(state: AgentState) -> dict:
        chunks = state.get("chunks", [])
        if not chunks:
            return {"relevant_chunks": [], "steps": ["No chunks to grade"]}
        # Keep one call each for generation and grounding; grading uses only the remainder.
        remaining_grading_calls = max(0, max_router_calls - state["router_calls"] - 2)
        if remaining_grading_calls == 0:
            return {"relevant_chunks": chunks, "steps": ["Skipped grading: reserved generation and grounding budget"]}
        selected = chunks[:remaining_grading_calls]
        relevant, calls = grade_chunks(state["question"], selected, router)
        # Chunks beyond the grading budget remain eligible evidence.
        relevant.extend(chunks[remaining_grading_calls:])
        return {"relevant_chunks": relevant, "router_calls": state["router_calls"] + calls,
                "steps": [f"Graded: {len(relevant)}/{len(chunks)} chunks retained"]}

    def generate(state: AgentState) -> dict:
        chunks = state.get("relevant_chunks", [])
        if not chunks:
            return {"draft_answer": "", "steps": ["No relevant chunks for generation"]}
        try:
            draft = generate_chain.invoke(
                {"context": format_context(chunks), "question": state["question"]}
            ).strip()
        except Exception:
            return {"draft_answer": "", "router_calls": state["router_calls"] + 1,
                    "stop_reason": "llm_unavailable", "steps": ["Generation failed"]}
        return {"draft_answer": draft, "router_calls": state["router_calls"] + 1,
                "steps": ["Generated draft"]}

    def ground(state: AgentState) -> dict:
        draft = state.get("draft_answer", "").strip()
        if not draft or draft.casefold() == ABSTENTION.casefold():
            return {"is_grounded": False, "feedback": "No answer to check",
                    "grounding_feedback": "No answer to check", "steps": ["Grounding skipped: no draft"]}
        if state["router_calls"] >= max_router_calls:
            feedback = "Grounding check skipped: router-call budget exhausted"
            return {"is_grounded": False, "feedback": feedback, "grounding_feedback": feedback,
                    "steps": ["Grounding skipped: budget exhausted"]}
        result = check_grounding(draft, state.get("relevant_chunks", []), router,
                                 grounded_threshold=grounded_threshold)
        status = "grounded" if result["is_grounded"] else "not grounded"
        return {"is_grounded": result["is_grounded"], "feedback": result["feedback"],
                "grounding_feedback": result["feedback"],
                "router_calls": state["router_calls"] + result["router_calls"],
                "steps": [f"Grounding check: {status}"]}

    def rewrite(state: AgentState) -> dict:
        query, calls = rewrite_query(state["question"], state.get("search_query", state["question"]),
                                     state.get("feedback", "No relevant results"),
                                     state.get("round", 0) + 1, router)
        return {"search_query": query, "round": state.get("round", 0) + 1,
                "router_calls": state["router_calls"] + calls, "steps": [f"Rewrote query: '{query}'"]}

    def abstain(state: AgentState) -> dict:
        reason = state.get("stop_reason") or (
            "out_of_scope" if state.get("route") == "abstain" else
            "budget_exhausted" if state["router_calls"] >= max_router_calls else
            "no_relevant_docs" if not state.get("relevant_chunks") else "insufficient_evidence")
        return {"final_answer": ABSTENTION, "citations": [], "status": "abstained",
                "stop_reason": reason, "steps": [f"Abstained: {reason}"]}

    def finalize(state: AgentState) -> dict:
        draft = state.get("draft_answer", "").strip()
        if not draft or draft.casefold() == ABSTENTION.casefold():
            return {"final_answer": ABSTENTION, "citations": [], "status": "abstained",
                    "stop_reason": state.get("stop_reason", "no_answer"), "steps": ["Finalized: abstained"]}
        cited = list(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_.:-]+)\]", draft)))
        valid = {c.payload["chunk_id"] for c in state.get("relevant_chunks", [])}
        if not cited or any(cid not in valid for cid in cited):
            return {"final_answer": ABSTENTION, "citations": [], "status": "abstained",
                    "stop_reason": "invalid_citations", "steps": ["Finalized: invalid citations"]}
        grounded = state.get("is_grounded", False)
        return {"final_answer": draft, "citations": cited,
                "status": "answered" if grounded else "answered_unverified",
                "stop_reason": "grounded" if grounded else "grounding_unverified",
                "steps": ["Finalized: answered"]}

    def after_plan(state):
        return state.get("route", "abstain")

    def after_grade(state):
        if not state.get("relevant_chunks"):
            return "rewrite" if state.get("round", 0) < max_rounds and state["router_calls"] + 3 <= max_router_calls else "abstain"
        # Preserve enough budget for generation and grounding.
        if state["router_calls"] + 1 > max_router_calls:
            return "abstain"
        return "generate"

    def after_ground(state):
        if state.get("is_grounded"):
            return "finalize"
        can_retry = state.get("round", 0) < max_rounds and state["router_calls"] + 3 <= max_router_calls
        return "rewrite" if can_retry else "finalize"

    builder = StateGraph(AgentState)
    for name, fn in [("initialize", initialize), ("plan", plan), ("retrieve", retrieve), ("grade", grade), ("generate", generate),
                     ("ground", ground), ("rewrite", rewrite), ("abstain", abstain), ("finalize", finalize)]:
        builder.add_node(name, fn)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "plan")
    builder.add_conditional_edges("plan", after_plan, {"retrieve": "retrieve", "abstain": "abstain"})
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", after_grade,
                                  {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"})
    builder.add_edge("generate", "ground")
    builder.add_conditional_edges("ground", after_ground, {"finalize": "finalize", "rewrite": "rewrite"})
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("abstain", END)
    builder.add_edge("finalize", END)
    return builder.compile()
