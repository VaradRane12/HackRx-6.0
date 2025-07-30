from fastapi import APIRouter
from app.schema import HackRxInput
from app.rag_pipeline import process_queries

router = APIRouter()

@router.post("/hackrx/run")
async def hackrx_run(req: HackRxInput):
    return await process_queries(req)
