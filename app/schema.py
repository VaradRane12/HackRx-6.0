from pydantic import BaseModel
from typing import List

class HackRxInput(BaseModel):
    documents: str  # PDF URL
    questions: List[str]
