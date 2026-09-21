from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
import os

load_dotenv()

llm = ChatMistralAI(
    model="open-mistral-7b",
    api_key=os.getenv("MISTRAL_API_KEY")
)

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


def generate_joke(state: JokeState) -> dict:
    print("STATE RECEIVED:", state)

    topic = state["topic"]

    prompt = f"Generate a joke about {topic}"
    response = llm.invoke(prompt).content

    return {"joke": response}


def explain_joke(state: JokeState) -> dict:
    joke = state["joke"]

    prompt = f"Explain this joke:\n{joke}"
    response = llm.invoke(prompt).content

    return {"explanation": response}


graph = StateGraph(JokeState)

graph.add_node("generate_joke", generate_joke)
graph.add_node("explain_joke", explain_joke)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "explain_joke")
graph.add_edge("explain_joke", END)

checkpoint = InMemorySaver()
app = graph.compile(checkpointer=checkpoint)

config1 = {
    "configurable": {
        "thread_id": "2"
    }
}

result = app.invoke(
    {"topic": "programming"},
    config=config1
)

print(result)
