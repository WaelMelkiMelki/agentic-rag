from langchain_core.prompts import ChatPromptTemplate

from agentic_rag.llm.chains import text_chain

REWRITE_PROMPT = """Your previous search did not find good results.
Original question: {question}
Previous query: {previous_query}
Round: {round}
Feedback: {feedback}
Generate a more specific search query using key terms, not a full sentence.
Respond with ONLY the new query (no explanation):"""

REWRITE_CHAT_PROMPT = ChatPromptTemplate.from_template(REWRITE_PROMPT)


def rewrite_query(question: str, previous_query: str, feedback: str, round_num: int, router) -> tuple[str, int]:
    chain = text_chain(REWRITE_CHAT_PROMPT, router)
    try:
        query = chain.invoke(
            {
                "question": question,
                "previous_query": previous_query,
                "feedback": feedback,
                "round": round_num,
            }
        ).strip()
        return (query if 3 <= len(query) <= 200 else question), 1
    except Exception:
        return question, 1
