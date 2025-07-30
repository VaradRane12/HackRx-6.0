from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)  # ✅ this needs actual Documents
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)
