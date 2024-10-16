import dotenv
import os

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool

dotenv.load_dotenv()
_DB_PATH = os.getenv('DB_PATH')

@tool
async def get_metadata(query_string: str) -> list[dict]:
    ''' Returns the pdf name and the page number which contains the text relevant to the query string.

    Args:
        query_string (str): The query string to search for in the vector database. 

    Returns:
        str: The pdf name and the page number which contains the text relevant to the query string.
    '''
    vector_store = _get_vector_store()
    results = await vector_store.asimilarity_search(query_string)
    
    text_locations = ''
    for result in results:
        file_name = result.metadata['file_path'].split(r"\\")[-1]
        page_number = result.metadata['page']
        text_locations += f"pdf_name: {file_name}, page_number: {page_number}\n"

    return text_locations

def _get_vector_store() -> Chroma:
    return Chroma(persist_directory=_DB_PATH, embedding_function=_get_embedding_model())

def _get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")