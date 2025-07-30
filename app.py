import os
import tempfile
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import messages_from_dict, messages_to_dict

from dotenv import load_dotenv
load_dotenv()  # make sure OPENAI_API_KEY is in your .env

app = FastAPI()

# ----------------------------
# INPUT/OUTPUT SCHEMAS
# ----------------------------
class HackRxInput(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# ----------------------------
# LOAD DOCS FROM PDF URL
# ----------------------------
def load_docs(url: str):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name
    loader = PyMuPDFLoader(tmp_path)
    return loader.load()

# ----------------------------
# BUILD VECTORSTORE
# ----------------------------
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)

# ----------------------------
# GET LLM
# ----------------------------
def get_llm():
    return ChatOpenAI(model_name="gpt-4", temperature=0)

# ----------------------------
# MAIN RAG PROCESS
# ----------------------------
def process_rag_pipeline(docs, questions):
    vectorstore = build_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)

    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        return_source_documents=False
    )

    results = []
    chat_history = []
    for q in questions:
        result = chain.invoke({"question": q, "chat_history": chat_history})
        results.append(result["answer"])
        chat_history.append(("user", q))
        chat_history.append(("ai", result["answer"]))
    return results

# ----------------------------
# API ROUTE
# ----------------------------
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(req: HackRxInput):
    docs = load_docs(req.documents)
    answers = process_rag_pipeline(docs, req.questions)
    return HackRxResponse(answers=answers)
