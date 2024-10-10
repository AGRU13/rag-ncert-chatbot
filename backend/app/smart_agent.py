import argparse
import asyncio
import dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from llm_with_rag import retreive_using_rag
from classifier import detect_outlier

dotenv.load_dotenv()

async def retrieve_using_smart_agent(query_text: str) -> str:
    is_outlier = await detect_outlier(query_text)
    llm = _get_chat_llm()

    if is_outlier:
        print("Outlier detected, using LLM")
        rag_chain = llm | StrOutputParser()
        return await rag_chain.ainvoke(query_text)
    else:
        print("Outlier not detected, using RAG")
        return await retreive_using_rag(query_text)

def _get_chat_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_retries=1,
    )

def _parse_arg() -> str:
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    return args.query_text

if __name__ == "__main__":
    query_text = _parse_arg()
    response = asyncio.run(retrieve_using_smart_agent(query_text))
    print(response)