from agentic_rag.agent.prompts import GENERATE_PROMPT, format_context
from agentic_rag.llm.router import LLMRouter
from agentic_rag.retrieval.store import VectorStore


REFUSAL_ANSWER = "I cannot answer this from the provided sources."
MIN_RETRIEVAL_SCORE = 0.3


class SimpleRAG:
    """Baseline RAG: retrieve top-k chunks, stuff context into a prompt, and generate."""

    def __init__(self, router=None, embedder=None, store=None, top_k: int = 5):
        self.router = router if router is not None else LLMRouter()
        if embedder is None:
            from agentic_rag.retrieval.embedder import CachedEmbedder

            embedder = CachedEmbedder()
        self.embedder = embedder
        self.store = store if store is not None else VectorStore()
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.top_k = top_k

    def retrieve(self, question: str, top_k: int = 5) -> list:
        """Search the vector store for chunks relevant to a question."""
        query_vector = self.embedder.embed_query(question)
        return self.store.search(query_vector, top_k=top_k)

    def generate(self, question: str, chunks) -> str:
        """Generate an answer using only the retrieved chunks."""
        context = format_context(chunks)
        prompt = GENERATE_PROMPT.format(context=context, question=question)
        return self.router.invoke_with_fallback(prompt)

    def ask(self, question: str, top_k: int = 5) -> dict:
        """Retrieve context and return a structured answer."""
        chunks = self.retrieve(question, top_k=top_k)

        if not chunks or chunks[0].score < MIN_RETRIEVAL_SCORE:
            return {
                "answer": REFUSAL_ANSWER,
                "sources": [],
                "retrieval_score": 0.0,
            }

        answer = self.generate(question, chunks)
        return {
            "answer": answer,
            "sources": [
                {
                    "chunk_id": chunk.payload["chunk_id"],
                    "doc_id": chunk.payload["doc_id"],
                    "score": round(chunk.score, 3),
                }
                for chunk in chunks
            ],
            "retrieval_score": round(chunks[0].score, 3),
        }

    def query(self, question: str) -> dict:
        """Return the baseline result in the API's answer/citations shape."""
        result = self.ask(question, top_k=self.top_k)
        citations = [source["chunk_id"] for source in result["sources"]]
        answered = bool(result["sources"])
        return {
            "answer": result["answer"],
            "citations": citations,
            "steps": ["retrieve", "generate"] if answered else ["retrieve", "abstain"],
            "status": "answered" if answered else "abstained",
            "stop_reason": "completed" if answered else "insufficient_retrieval_score",
        }
