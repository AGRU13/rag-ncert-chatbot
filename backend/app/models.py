from pydantic import BaseModel

class QueryInput(BaseModel):
    query: str

class QueryOutput(BaseModel):
    answer: str