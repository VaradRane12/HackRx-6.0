from app.vectorstore import build_vectorstore
from app.llm_utils import get_llm
from langchain.chains import ConversationalRetrievalChain
from app.document_loader import load_docs
from app.llm_utils import get_llm

chat_history = []  # For simplicity, this is empty. You can add previous Q&A pairs if needed.

async def process_queries(req):
    docs =req.documents
    questions = req.questions

    # Build vectorstore and retriever
    vectorstore = build_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Load your LLM
    llm = get_llm()

    # Set up ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # Needed for explainability
    )

    # Generate answers with traceability
    answers = []
    for q in questions:
        result = chain.invoke({"question": q, "chat_history": chat_history})
        source_docs = result.get("source_documents", [])
        source_clauses = [doc.page_content[:300] for doc in source_docs]  # Limit length to avoid token bloat

        answers.append({
            "question": q,
            "answer": result["answer"],
            "clauses": source_clauses,
            "notes": "Answer generated from top-matching policy clauses."
        })

    return {"answers": answers}
