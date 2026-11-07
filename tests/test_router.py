"""Manual Gemini connectivity check; run with ``python tests/test_router.py``."""


def main():
    from agentic_rag.llm.router import LLMRouter

    router = LLMRouter()
    response = router.get_llm("gemini").invoke("What is 2+2?")
    print(response.content)


if __name__ == "__main__":
    main()
