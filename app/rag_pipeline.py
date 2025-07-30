from app.document_loader import load_pdf_from_url
from app.vectorstore import build_vectorstore
from app.llm_utils import build_chain

async def process_queries(req):
    docs = load_pdf_from_url(req.documents)
    vectorstore = build_vectorstore(docs)
    chain = build_chain(vectorstore)

    answers = []
    chat_history = []
    for q in req.questions:
        result = chain({"question": q, "chat_history": chat_history})
        answers.append(result["answer"])
    return {"answers": answers}
