from typing import Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    action: Literal["retrieve", "abstain"]
    query: str = Field(default="", max_length=400)


PLAN_PROMPT = """You are routing questions for a document-QA system. The document collection contains research papers about Transformers, retrieval-augmented generation, and related AI topics.
Decide:
- "retrieve": the question might be answered using these papers.
- "abstain": the question is clearly unrelated to these papers.
If unsure, choose "retrieve". Do not answer the question yourself.
Return ONLY JSON in this shape:
{{"action": "retrieve", "query": "a concise search query"}}
For "abstain", the query may be empty.

Question: {question}
"""
