import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict


class ClaimCheck(TypedDict):
    claim: str
    verdict: Literal["supported", "partial", "unsupported"]
    chunk_ids: list[str]


class AgentState(TypedDict):
    # Supplied when invoking the graph
    question: str
    router_calls: int
    steps: Annotated[list[str], operator.add]

    # Filled by graph nodes
    route: NotRequired[Literal["retrieve", "abstain"]]
    search_query: NotRequired[str]
    # ``retrieval.retriever.RetrievedChunk`` — kept as Any to avoid a Qdrant import here.
    chunks: NotRequired[list[Any]]
    round: NotRequired[int]
    relevant_chunks: NotRequired[list[Any]]
    draft_answer: NotRequired[str]
    is_grounded: NotRequired[bool]
    grounding_feedback: NotRequired[str]
    grounding_claims: NotRequired[list[ClaimCheck]]
    feedback: NotRequired[str]
    final_answer: NotRequired[str]
    citations: NotRequired[list[str]]
    status: NotRequired[Literal["answered", "answered_unverified", "abstained"]]
    stop_reason: NotRequired[str]
