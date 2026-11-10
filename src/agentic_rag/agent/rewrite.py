from langchain_core.prompts import ChatPromptTemplate

REWRITE_PROMPT = """Your previous search did not find good results.
Original question: {question}
Previous query: {previous_query}
Round: {round}
Feedback: {feedback}
Generate a more specific search query using key terms, not a full sentence.
Respond with ONLY the new query (no explanation):"""

REWRITE_CHAT_PROMPT = ChatPromptTemplate.from_template(REWRITE_PROMPT)


def rewrite_query(question: str, previous_query: str, feedback: str, round_num: int, router) -> tuple[str, int]:
    messages = REWRITE_CHAT_PROMPT.format_messages(
        question=question, previous_query=previous_query,
        feedback=feedback, round=round_num)
    try:
        query = router.invoke_with_fallback(messages).strip()
        return (query if 3 <= len(query) <= 200 else question), 1
    except Exception:
        return question, 1
