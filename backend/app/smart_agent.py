import argparse
import asyncio
import dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from llm_with_rag import retreive_using_rag
from classifier import detect_outlier
from get_metadata import get_metadata

dotenv.load_dotenv()

tools = [get_metadata, retreive_using_rag]

async def retrieve_using_smart_agent(query_text: str) -> str:
    is_outlier = await detect_outlier(query_text)
    llm = _get_chat_llm()
    llm_with_tools = llm.bind_tools(tools)

    if is_outlier:
        print("Outlier detected, using LLM")
        chain = llm | StrOutputParser()
        return await chain.ainvoke(query_text)
    else:
        print("Outlier not detected, using RAG")

        messages = [HumanMessage(query_text)]
        ai_msg = await llm_with_tools.ainvoke(messages)
        print("tool_calls: ", ai_msg.tool_calls)
        messages.append(ai_msg)

        for tool_call in ai_msg.tool_calls:
            tool_name = {"get_metadata": get_metadata, "retreive_using_rag": retreive_using_rag}[tool_call['name'].lower()]
            tool_call_msg = await tool_name.ainvoke(tool_call)
            messages.append(tool_call_msg)

        chain = llm_with_tools | StrOutputParser()
        return await chain.ainvoke(messages)
                

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