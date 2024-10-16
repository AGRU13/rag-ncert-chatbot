import argparse
import dotenv
import logging
import os
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.tools import tool
dotenv.load_dotenv()
_DB_PATH = os.getenv('DB_PATH')

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

@tool
async def retreive_using_rag(query_text: str) -> str:
    '''
    Retrieves the most relevant documents from the vector database and uses them to answer the query.

    Args:
        query_text (str): The query text to search for in the vector database.

    Returns:
        str: The answer to the query.
    '''

    retriever = _get_vector_store().as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | _get_chat_llm()
        | StrOutputParser()
    )
    return await rag_chain.ainvoke(query_text)

def _get_vector_store() :
    return Chroma(persist_directory=_DB_PATH, embedding_function=_get_embedding_model())

def _get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def _get_chat_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_retries=1,
    )

def _format_docs(docs):
    # print([doc.metadata for doc in docs])
    return "\n\n".join(doc.page_content for doc in docs)

def _parse_arg() -> str:
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    return args.query_text

if __name__ == "__main__":
    query_text = _parse_arg()
    response = asyncio.run(retreive_using_rag(query_text))
    print(response)