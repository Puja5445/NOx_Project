import pandas as pd # nox excel data for model building

df = pd.read_excel('Data/Furnace7_NOx_data.xlsx')

print(df.head())
print(df.shape)
print(df.columns)


#=============================================================================================
# nox documentation for RAG building

from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

doc_text = read_docx("Data/NOx_document_F7.docx")

print(doc_text[0:1000])

#clean text Makes text consistent and Helps embedding later
def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    return text

cleaned_text = clean_text(doc_text)

print('clean text: ',cleaned_text[:500])

#Split into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

print("Starting chunking...")

chunks = text_splitter.split_text(cleaned_text)

print("Chunking done")

print('length of chunks: ', len(chunks))
print('chunking index:', chunks[0])

# Now the chunking is done, will move to embeddings means convert the text /chunks into vectors[set of numbers]

import os
os.environ["OPENAI_API_KEY"] = "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"
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Create embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create FAISS vector store directly
vector_store = FAISS.from_texts(chunks, embeddings)

print("FAISS created with OpenAI embeddings")

# Ask question to the system now
query = "What factors affect NOx emissions?"

# Search top 3 relevant chunks
docs = vector_store.similarity_search(query, k=3)

print("Top results:\n")

for i, doc in enumerate(docs):
    print(f"Result {i+1}:")
    print(doc.page_content)
    print("--------")

# Augumentation + LLM --> Final Answer

from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

query = "What factors affect NOx emissions?"

# Retrieve top chunks
docs = vector_store.similarity_search(query, k=3)

# Combine retrieved text
context = "\n".join([doc.page_content for doc in docs])

# Create prompt -- normal prompt
'''prompt = f"""
You are an expert in NOx emission modeling.

Answer the question based only on the context below. If you dont know the answer just say I dont know.

Context:
{context}

Question:
{query}
"""'''

# prompt improvement to reduce halluciantion -- new prompt
prompt = f"""
You are a chemical process expert specializing in NOx emissions.

Instructions:
- Answer ONLY from the provided context
- Be clear and concise
- If answer is not available, say "I don't know"

Context:
{context}

Question:
{query}
"""


# Get response
#response = llm.invoke(prompt)

#print("Final Answer:\n")
#print(response.content)


# now will test the RAG with multiple queries to see if it answers well for all the questions asked

queries = [
    "What factors affect NOx emissions?",
    "How does temperature influence NOx?",
    "Why was SVM used in this model?",
    "What variables impact combustion efficiency?",
    "How can NOx emissions be reduced?"
]

for query in queries:
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a chemical process expert specializing in NOx emissions.

    Answer ONLY from the context below.
    If answer is not present, say "I don't know".

    Context:
    {context}

    Question:
    {query}
    """

    #response = llm.invoke(prompt)

    #print("\n-----------------------------")
    #print("Question:", query)
    #print("Answer:", response.content)

    # above is my RAG manual code, Now I will building the pipeline using Langchain Runnable which is reusable code hence below is the 
    # runnable code for this RAG.

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Prompt template
prompt_template = PromptTemplate.from_template("""
You are a chemical process expert specializing in NOx emissions.

Answer ONLY from the context below.
If the answer is not available, say "I don't know".

Context:
{context}

Question:
{question}
""")

# Function to format retrieved docs
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

# Runnable pipeline
rag_chain = (
    {
        "context": lambda x: format_docs(vector_store.similarity_search(x, k=3)),
        "question": lambda x: x
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# asking query after running the pipeline through runnable
query = "What factors affect NOx emissions?"

#response = rag_chain.invoke(query)

#print("\nRunnable Answer:")
#print(response)

#===============================================================================================================================================#

# Now after this RAG manual and pipeline mode, I am moving towards to ML model building using SVM to do NOx Forecasting
# =========================
# SVM NOx Prediction Model
# =========================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_excel("Data/Furnace7_NOx_data.xlsx")

# Features
X = df[[
    "Fuel_gas_flow_to_upper_Burner",
    "Bridgewall_Temp_Avg",
    "Adiabatic Flame Temperature"
]]

# Target
y = df["Furnace7_Nox_In_Ng_Per_J"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = SVR(kernel="rbf")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
train_pred = model.predict(X_train)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, y_pred)

print("Train MAE:", train_mae)
print("Test MAE:", test_mae)

# Manual prediction (IMPORTANT)
import pandas as pd

new_input = pd.DataFrame([{
    "Fuel_gas_flow_to_upper_Burner": 3200,
    "Bridgewall_Temp_Avg": 1350,
    "Adiabatic Flame Temperature": 2250
}])

predicted_nox = model.predict(new_input)

print("Predicted NOx:", predicted_nox[0])

#=============================================================================================================================================#
# now we are going to build the Explainable AI system, which means we will write a function to add ML + RAG where ML will give the number and
# RAG will give the answer

def predict_and_explain(fuel, temp, aft):
    
    import pandas as pd

    # ---- Step 1: ML Prediction ----
    input_data = pd.DataFrame([{
        "Fuel_gas_flow_to_upper_Burner": fuel,
        "Bridgewall_Temp_Avg": temp,
        "Adiabatic Flame Temperature": aft
    }])

    predicted_nox = model.predict(input_data)[0]

    # ---- Step 2: RAG Explanation ----
    query = f"""
    Explain why NOx is affected when:
    Fuel gas flow = {fuel},
    Temperature = {temp},
    Adiabatic flame temperature = {aft}
    """

    explanation = rag_chain.invoke(query)

    # ---- Step 3: Combine ----
    final_output = f"""
    Predicted NOx: {predicted_nox:.2f}

    Explanation:
    {explanation}
    """

    return final_output


result = predict_and_explain(3200, 1350, 2250)

print(result)
#=========================================================================================================================================#
# Now the next big step is to build Agentic AI using LangGraph which has State, nodes and edges, A decision-making system (agent):
#'''User input
  # ↓
#Agent decides:
  # → Numeric → ML (SVM)
   #→ Conceptual → RAG'''

# Defining function for state
from typing import TypedDict

class State(TypedDict):
    input: str
    output: str


# Node1 - Decision Function

def decide(state):
    user_input = state["input"]

    # Simple rule (we can improve later)
    if any(char.isdigit() for char in user_input):
        return {"next": "ml"}
    else:
        return {"next": "rag"}
    
# Node 2 - ML Node
def ml_node(state):
    # Example parsing (simple)
    # Assume input like: "3200 1350 2250"

    values = list(map(float, state["input"].split()))

    fuel, temp, aft = values

    result = predict_and_explain(fuel, temp, aft)

    return {"output": result}

# Node 3 - RAG Node
def rag_node(state):
    response = rag_chain.invoke(state["input"])
    return {"output": response}

# Build Graph 
from langgraph.graph import StateGraph, END

graph = StateGraph(State)

# Add nodes
graph.add_node("decide", decide)
graph.add_node("ml", ml_node)
graph.add_node("rag", rag_node)

# Entry point
graph.set_entry_point("decide")

# Conditional edges
graph.add_conditional_edges(
    "decide",
    lambda x: x["next"],
    {
        "ml": "ml",
        "rag": "rag"
    }
)

# End nodes
graph.add_edge("ml", END)
graph.add_edge("rag", END)

# Compile
app = graph.compile()

# Run Agent 
#result = app.invoke({"input": "3200 1350 2250"})
#print(result["output"])

# Try RAG input 

if __name__ == "__main__":
    
    # Test RAG
    result = app.invoke({"input": "What affects NOx emissions?"})
    print(result["output"])

    # Graph (optional)
    # app.get_graph().draw_png("graph.png")

    # Test rag_chain
    response = rag_chain.invoke("test query")
    print(response)


# Visual graph of the conditional workflow

#app.get_graph().draw_png("graph.png")







