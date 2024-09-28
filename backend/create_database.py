import os
import dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

dotenv.load_dotenv()

_DATA_PATH = 'data'
_DB_PATH = os.getenv('DB_PATH')

def main():
    embedding_model = _get_embedding_model()
    _create_vector_datastore(embedding_model)

def _get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def _create_vector_datastore(embedding_model: Embeddings):
    documents = _load_pdf_documents()
    chunks = _split_text(documents)
    _save_to_db(chunks, embedding_model)

def _load_pdf_documents():
    loader = DirectoryLoader(_DATA_PATH, glob="*.pdf", use_multithreading=True, loader_cls=PyMuPDFLoader)
    return loader.load()

def _split_text(documents: list[Document]) -> list[Document]:
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

def _save_to_db(chunks: list[Document], embedding_model: Embeddings):
    vector_store = Chroma.from_documents(chunks, embedding_model, persist_directory=_DB_PATH)
    vector_store.reset_collection()

    vector_store.add_documents(documents=chunks)
    print(f"Saved {len(chunks)} chunks to {_DB_PATH}.")
    return vector_store

if __name__ == "__main__":
    main()