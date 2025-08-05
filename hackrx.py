import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec, PineconeVectorStore

# Load keys from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# PDF loading
loader = PyPDFLoader("pdf.pdf")
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Embeddings
embedding = OpenAIEmbeddings()

# Pinecone v3 client setup
pc = PineconeClient(api_key=PINECONE_API_KEY)

index_name = "pdf-vector-index"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(index_name)

# Use LangChain Pinecone wrapper
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Query
query = "does this cover knee injury?"
retrieved_results = db.similarity_search(query)
print(retrieved_results[0].page_content)
