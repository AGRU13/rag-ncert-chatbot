from llm_with_rag import retreive_using_rag
from models import QueryInput, QueryOutput 

from fastapi import FastAPI

app = FastAPI(
    title="NCERT Chatbot", 
    descrition="Endpoints for rag chatbot ealing with NCERT books.",
)

@app.get("/")
async def get_statuc():
    return {'statuc': 'running'}

@app.post("/ncert-chatbot")
async def query_llm(request: QueryInput) -> QueryOutput:
    response = await retreive_using_rag(request.query)
    return {'answer': response}