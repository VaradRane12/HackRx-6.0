import aiohttp
import tempfile
import os
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "local_faiss_index"

async def download_pdf(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception("Failed to download PDF")
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp.write(await resp.read())
            temp.close()
            return temp.name

async def process_pdf_and_answer(pdf_path: str, questions: list) -> list:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    if os.path.exists(INDEX_DIR):
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_DIR)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0),
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    # Using invoke safely
    responses = []
    for q in questions:
        result = qa.invoke({"query": q})
        if isinstance(result, dict):
            responses.append(result.get("result", str(result)))
        else:
            responses.append(str(result))
    return responses
