import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from helper import load_pdf_documents, split_text

_DB_PATH = os.getenv('DB_PATH')

def main():
    embedding_model = _get_embedding_model()
    _create_vector_datastore(embedding_model)

def _get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def _create_vector_datastore(embedding_model: Embeddings):
    documents = load_pdf_documents()
    chunks = split_text(documents)
    _save_to_db(chunks, embedding_model)

def _save_to_db(chunks: list[Document], embedding_model: Embeddings):
    vector_store = Chroma.from_documents(chunks, embedding_model, persist_directory=_DB_PATH)
    vector_store.reset_collection()

    vector_store.add_documents(documents=chunks)
    print(f"Saved {len(chunks)} chunks to {_DB_PATH}.")

if __name__ == "__main__":
    main()