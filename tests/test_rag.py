"""Manual live RAG demo; run with ``python tests/test_rag.py``."""


def main():
    from agentic_rag.agent.rag import SimpleRAG

    rag = SimpleRAG()
    print("=" * 60)
    print("TEST 1: Answerable question")
    print("=" * 60)
    result = rag.ask("What is the purpose of multi-head attention in the Transformer?")
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Top retrieval score: {result['retrieval_score']}")
    print("Sources:")
    for source in result["sources"]:
        print(f"   [{source['chunk_id']}] score={source['score']}")

    print("\n" + "=" * 60)
    print("TEST 2: Another answerable question")
    print("=" * 60)
    result = rag.ask("How does RAG improve LLM performance?")
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Top retrieval score: {result['retrieval_score']}")

    print("\n" + "=" * 60)
    print("TEST 3: Unanswerable question (should refuse)")
    print("=" * 60)
    result = rag.ask("What is the best pizza recipe in New York?")
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Top retrieval score: {result['retrieval_score']}")


if __name__ == "__main__":
    main()
