# NEW utils.py (FASTER + CACHED + BATCHED)
import aiohttp
import tempfile
import os
import hashlib
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR_BASE = "faiss_indexes"
os.makedirs(INDEX_DIR_BASE, exist_ok=True)

def get_pdf_hash(pdf_path):
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

async def download_pdf(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception("Failed to download PDF")
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp.write(await resp.read())
            temp.close()
            return temp.name

def process_pdf_and_answer(pdf_path: str, questions: list) -> list:
    pdf_hash = get_pdf_hash(pdf_path)
    index_dir = os.path.join(INDEX_DIR_BASE, pdf_hash)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    if os.path.exists(index_dir):
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_dir)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0),
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    # Run each question individually and extract the answer only
    return [
        qa.invoke({"query": q}).get("result", "") for q in questions
    ]
