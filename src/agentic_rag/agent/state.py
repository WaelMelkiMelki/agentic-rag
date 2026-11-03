import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from qdrant_client.models import ScoredPoint


class AgentState(TypedDict):
    # Supplied when invoking the graph
    question: str
    router_calls: int
    steps: Annotated[list[str], operator.add]

    # Filled by graph nodes
    route: NotRequired[Literal["retrieve", "abstain"]]
    search_query: NotRequired[str]
    chunks: NotRequired[list[ScoredPoint]]
    draft_answer: NotRequired[str]
    final_answer: NotRequired[str]
    citations: NotRequired[list[str]]
    status: NotRequired[Literal["answered_unverified", "abstained"]]
    stop_reason: NotRequired[str]
