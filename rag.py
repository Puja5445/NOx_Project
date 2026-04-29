from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Load doc
def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

doc_text = read_docx("Data/NOx_document_F7.docx")

# Clean text
def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    return text

cleaned_text = clean_text(doc_text)

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(cleaned_text)

# Embeddings + FAISS
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_texts(chunks, embeddings)

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Prompt
prompt_template = PromptTemplate.from_template("""
You are a chemical process expert specializing in NOx emissions.

Answer ONLY from the context below.
If the answer is not available, say "I don't know".

Context:
{context}

Question:
{question}
""")

# Format docs
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

# RAG chain
rag_chain = (
    {
        "context": lambda x: format_docs(vector_store.similarity_search(x, k=3)),
        "question": lambda x: x
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# Function to expose
def get_rag_answer(query):
    return rag_chain.invoke(query)