from fastapi import FastAPI
from app.routes import router
import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")
app = FastAPI()
app.include_router(router, prefix="/api/v1")
