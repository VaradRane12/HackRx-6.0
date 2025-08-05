import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone

# Load API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1-aws"
INDEX_NAME = "pdf-vector-index1"

# Step 1: Load PDF and split
loader = PyPDFLoader("pdf.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Step 2: Setup Embeddings
embeddings = OpenAIEmbeddings()

# Step 3: Init Pinecone (v2.x style)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Step 4: Create index if not exists
print(pinecone.list_indexes())
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=1536,  # must match OpenAI embedding size
        metric="cosine"
    )

# Step 5: Load into vectorstore
vectorstore = Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)

# Step 6: Search
query = "does this cover heart injury?"
results = vectorstore.similarity_search(query, k=1)

# Step 7: Show result
print("\n--- Top Matching Page ---\n")
print(results[0].page_content)
