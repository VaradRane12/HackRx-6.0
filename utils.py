import aiohttp
import tempfile
import os
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone

from dotenv import load_dotenv
load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1-aws"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "pdf-vector-index1"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

def get_index():
    return pinecone.Index(INDEX_NAME)

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
    vectorstore = LangchainPinecone.from_documents(chunks, embeddings, index_name=INDEX_NAME)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(api_key=OPENAI_API_KEY, temperature=0),
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    return [qa.run(q) for q in questions]
