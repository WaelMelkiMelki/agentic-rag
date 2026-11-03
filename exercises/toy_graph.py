from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph


class ToyState(TypedDict):
    count: int
    history: Annotated[list[str], operator.add]


def increment(state: ToyState) -> dict:
    new_count = state["count"] + 1
    return {
        "count": new_count,
        "history": [f"Count became {new_count}"],
    }


def decide(state: ToyState) -> str:
    return "again" if state["count"] < 3 else "done"


builder = StateGraph(ToyState)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_conditional_edges(
    "increment",
    decide,
    {"again": "increment", "done": END},
)

app = builder.compile()
result = app.invoke({"count": 0, "history": []})
print(result)
