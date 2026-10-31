import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class MovieRecommendation(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    reason: str = Field(description="Why recommend it", max_length=100)
    rating: float = Field(description="Rating out of 10", ge=0, le=10)


llm = ChatOpenAI(
    model="openrouter/free",
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)

structured_llm = llm.with_structured_output(MovieRecommendation)
result = structured_llm.invoke("Recommend a sci-fi movie")

print(result)
print(f"Title: {result.title}")
print(f"Year type: {type(result.year)}")
