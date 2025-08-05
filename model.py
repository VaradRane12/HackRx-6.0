from pydantic import BaseModel
from typing import List

class HackrxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackrxResponse(BaseModel):
    answers: List[str]
