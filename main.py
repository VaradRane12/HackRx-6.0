from fastapi import FastAPI, Header, HTTPException
from model import HackrxRequest, HackrxResponse
from utils import download_pdf, process_pdf_and_answer
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

API_KEY = os.getenv("BEARER_TOKEN")

@app.post("/hackrx/run", response_model=HackrxResponse)
async def run_hackrx(
    payload: HackrxRequest,
    authorization: str = Header(...)
):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")

    token = authorization.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

    try:
        pdf_path = await download_pdf(payload.documents)
        answers = await process_pdf_and_answer(pdf_path, payload.questions)
        os.remove(pdf_path)
        return HackrxResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
