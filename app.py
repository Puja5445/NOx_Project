from fastapi import FastAPI
from agent import run_agent
from ml_model import predict_nox
from rag import get_rag_answer

app = FastAPI()

# Home endpoint
@app.get("/")
def home():
    return {"message": "NOx AI system running"}

# RAG endpoint
@app.post("/ask")
def ask(question: str):
    return {"answer": get_rag_answer(question)}

# ML + Agent endpoint
@app.post("/predict")
def predict(fuel: float, temp: float, aft: float):
    prediction = predict_nox(fuel, temp, aft)
    explanation = run_agent(f"{fuel} {temp} {aft}")

    return {
        "prediction": prediction,
        "explanation": explanation
    }