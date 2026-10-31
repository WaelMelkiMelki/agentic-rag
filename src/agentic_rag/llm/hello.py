import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# OpenRouter uses the OpenAI interface with a custom base_url.
llm = ChatOpenAI(
    model="openrouter/free",
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)

response = llm.invoke("Who is the first prsident of the US ? ")
print(response.content)
llm2 = ChatOpenAI(
    model="openrouter/free",
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)
response2 = llm2.invoke("What is RAG in 2 sentences?")
print(response2.content)