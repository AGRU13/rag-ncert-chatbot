import os
import dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()
_DATA_PATH = os.getenv('DATA_PATH')

def load_pdf_documents():
    loader = DirectoryLoader(_DATA_PATH, glob="*.pdf", use_multithreading=True, loader_cls=PyMuPDFLoader)
    return loader.load()

def split_text(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    sum = 0
    for chunk in chunks:
        sum += len(chunk.page_content)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks with total text length {sum}")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks
