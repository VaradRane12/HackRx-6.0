from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

def build_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)
    return ConversationalRetrievalChain.from_llm(llm, retriever=retriever, return_source_documents=True)
