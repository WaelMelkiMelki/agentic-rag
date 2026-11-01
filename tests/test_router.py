from agentic_rag.llm.router import LLMRouter

router = LLMRouter()

# Call Gemini directly (skip OpenRouter)
llm = router.get_llm("gemini")
response = llm.invoke("What is 2+2?")
print(response.content)