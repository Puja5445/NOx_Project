from typing import TypedDict
from langgraph.graph import StateGraph, END

from rag import get_rag_answer
from ml_model import predict_nox

# State
class State(TypedDict):
    input: str
    output: str

# Decision node
def decide(state):
    user_input = state["input"]

    if any(char.isdigit() for char in user_input):
        return {"next": "ml"}
    else:
        return {"next": "rag"}

# ML node
def ml_node(state):
    values = list(map(float, state["input"].split()))
    fuel, temp, aft = values

    prediction = predict_nox(fuel, temp, aft)

    return {"output": f"Predicted NOx: {prediction:.2f}"}

# RAG node
def rag_node(state):
    response = get_rag_answer(state["input"])
    return {"output": response}

# Build graph
graph = StateGraph(State)

graph.add_node("decide", decide)
graph.add_node("ml", ml_node)
graph.add_node("rag", rag_node)

graph.set_entry_point("decide")

graph.add_conditional_edges(
    "decide",
    lambda x: x["next"],
    {"ml": "ml", "rag": "rag"}
)

graph.add_edge("ml", END)
graph.add_edge("rag", END)

graph_app = graph.compile()

# Function for FastAPI
def run_agent(user_input: str):
    return graph_app.invoke({"input": user_input})["output"]