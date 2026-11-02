from agentic_rag.agent.rag import SimpleRAG


rag = SimpleRAG()

# Break Gemini on purpose.
rag.router.gemini = None

print("Testing with Gemini broken (should use OpenRouter fallback)...\n")
result = rag.ask("What is self-attention in the Transformer?")
print(f"Retrieval score: {result['retrieval_score']}")

if not result["sources"]:
    raise RuntimeError("Fallback test did not retrieve any relevant sources")

if result["answer"] == "I cannot answer this from the provided sources.":
    raise RuntimeError("Fallback test returned the retrieval refusal instead of an answer")

print(f"Answer: {result['answer'][:200]}...")
print("\nFallback worked!")
