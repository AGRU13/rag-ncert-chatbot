import dotenv
import numpy as np

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings

from helper import load_pdf_documents, split_text

dotenv.load_dotenv()

def main():
    embedding_model = _get_embedding_model()
    _create_classifier(embedding_model)

def _get_embedding_model() -> Embeddings:
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="clustering")

def _create_classifier(embedding_model: Embeddings) -> None:
    documents = load_pdf_documents()
    chunks = split_text(documents)
    texts = [doc.page_content for doc in chunks]

    embeddings = embedding_model.embed_documents(texts)
    embeddings_array = np.array(embeddings)
    with open('embeddings.npy', 'wb') as f:
        np.save(f, embeddings_array)
    print(f"Saved {len(texts)} texts as array of size {embeddings_array.shape} to embeddings.npy")

if __name__ == "__main__":
    main()