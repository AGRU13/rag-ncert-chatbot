from llm_with_rag import retreive_using_rag
from smart_agent import retrieve_using_smart_agent
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

@app.post("/smart-agent")
async def query_smart_agent(request: QueryInput) -> QueryOutput:
    response = await retrieve_using_smart_agent(request.query)
    return {'answer': response}
