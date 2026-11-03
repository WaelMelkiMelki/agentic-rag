import re

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from agentic_rag.agent.plan import PLAN_PROMPT, Plan
from agentic_rag.agent.prompts import GENERATE_PROMPT, format_context
from agentic_rag.agent.state import AgentState

ABSTENTION = "I cannot answer this from the provided sources."


def build_agent(router, embedder, store, *, top_k: int = 5, max_router_calls: int = 2):
    if top_k < 1 or max_router_calls < 1:
        raise ValueError("top_k and max_router_calls must be positive")

    def plan(state: AgentState) -> dict:
        """Decide whether to retrieve and prepare a search query."""
        if state["router_calls"] >= max_router_calls:
            return {
                "route": "abstain",
                "stop_reason": "budget_exhausted",
                "steps": ["Planner skipped: budget exhausted"],
            }
        try:
            raw = router.invoke_with_fallback(
                PLAN_PROMPT.format(question=state["question"])
            )
        except Exception:
            return {
                "route": "abstain",
                "router_calls": state["router_calls"] + 1,
                "stop_reason": "llm_unavailable",
                "steps": ["Planner failed: LLM unavailable"],
            }

        try:
            parsed = Plan.model_validate_json(raw.strip())
            route = parsed.action
            query = parsed.query.strip() or state["question"]
            step = f"Planner chose: {route}"
        except ValidationError:
            # A malformed plan must not cause an uncontrolled action.
            # Searching the read-only document store is our safe default.
            route = "retrieve"
            query = state["question"]
            step = "Planner output invalid; searched original question"

        return {
            "route": route,
            "search_query": query,
            "router_calls": state["router_calls"] + 1,
            "steps": [step],
        }

    def retrieve(state: AgentState) -> dict:
        """Embed the planned query and search Qdrant. No router call."""
        try:
            vector = embedder.embed_query(state["search_query"])
            chunks = store.search(vector, top_k=top_k)
        except Exception:
            return {
                "chunks": [],
                "stop_reason": "retrieval_unavailable",
                "steps": ["Retrieval failed"],
            }
        return {
            "chunks": chunks,
            "steps": [f"Retrieved {len(chunks)} chunks"],
        }

    def generate(state: AgentState) -> dict:
        """Use the SAME generation prompt as the B1 baseline."""
        context = format_context(state["chunks"])
        prompt = GENERATE_PROMPT.format(
            context=context,
            question=state["question"],
        )
        try:
            draft = router.invoke_with_fallback(prompt)
        except Exception:
            return {
                "draft_answer": "",
                "router_calls": state["router_calls"] + 1,
                "stop_reason": "llm_unavailable",
                "steps": ["Generation failed: LLM unavailable"],
            }
        return {
            "draft_answer": draft.strip(),
            "router_calls": state["router_calls"] + 1,
            "steps": ["Generated a draft answer"],
        }

    def abstain(state: AgentState) -> dict:
        if state.get("stop_reason"):
            reason = state["stop_reason"]
        elif state.get("route") == "abstain":
            reason = "out_of_scope"
        elif state["router_calls"] >= max_router_calls:
            reason = "budget_exhausted"
        else:
            reason = "no_chunks"

        return {
            "final_answer": ABSTENTION,
            "citations": [],
            "status": "abstained",
            "stop_reason": reason,
            "steps": [f"Abstained: {reason}"],
        }

    def finalize(state: AgentState) -> dict:
        """Check citation IDs exist. Do NOT claim this proves grounding."""
        draft = state.get("draft_answer", "").strip()
        if not draft or draft.casefold() == ABSTENTION.casefold():
            return {
                "final_answer": ABSTENTION,
                "citations": [],
                "status": "abstained",
                "stop_reason": state.get("stop_reason", "insufficient_evidence"),
                "steps": ["Abstained after generation"],
            }

        cited_ids = list(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_.:-]+)\]", draft)))
        available_ids = {
            chunk.payload["chunk_id"] for chunk in state["chunks"]
        }

        if not cited_ids or any(
            chunk_id not in available_ids for chunk_id in cited_ids
        ):
            return {
                "final_answer": ABSTENTION,
                "citations": [],
                "status": "abstained",
                "stop_reason": "invalid_or_missing_citations",
                "steps": ["Rejected answer: invalid or missing citations"],
            }

        return {
            "final_answer": draft,
            "citations": cited_ids,
            "status": "answered_unverified",
            "stop_reason": "completed",
            "steps": ["Finalized answer; grounding not yet checked"],
        }

    def after_plan(state: AgentState) -> str:
        return state["route"]

    def after_retrieve(state: AgentState) -> str:
        if not state.get("chunks"):
            return "abstain"
        if state["router_calls"] >= max_router_calls:
            return "abstain"
        return "generate"

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("abstain", abstain)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan",
        after_plan,
        {"retrieve": "retrieve", "abstain": "abstain"},
    )
    builder.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {"generate": "generate", "abstain": "abstain"},
    )
    builder.add_edge("generate", "finalize")
    builder.add_edge("abstain", END)
    builder.add_edge("finalize", END)

    return builder.compile()
