import argparse
import asyncio
import numpy as np

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings

def _load_classifier() -> np.ndarray:
    with open('./../utilities/embeddings.npy', 'rb') as f:
        embeddings_array = np.load(f)
    print(f"loaded embeddings array of size {embeddings_array.shape} from embeddings.npy")
    return embeddings_array

def _get_embedding_centroid(embeddings_array: np.ndarray):
     # Get the centroid value of dimension 768
    return np.mean(embeddings_array, axis=0)

async def _get_embedding(query_text: str) -> np.ndarray:
    embedding_model = _get_embedding_model()
    return await embedding_model.aembed_query(query_text)

def _get_embedding_model() -> Embeddings:
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="clustering")

def _detect_outlier(embedding_array: np.ndarray, embedding_centroid: np.ndarray, radius: float):
    dist = _calculate_euclidean_distance(embedding_array, embedding_centroid)
    print(f"Distance: {dist}")
    return dist > radius

def _calculate_euclidean_distance(p1: np.ndarray, p2: np.ndarray):
    return np.sqrt(np.sum(np.square(p1 - p2)))

classifier_embeddings_array = _load_classifier()
embedding_centroid = _get_embedding_centroid(classifier_embeddings_array)

async def detect_outlier(query_text: str) -> bool:
    query_embedding_array = await _get_embedding(query_text)
    return _detect_outlier(query_embedding_array, embedding_centroid, 0.9)

def _parse_arg() -> str:
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    return args.query_text

if __name__ == "__main__":
    query_text = _parse_arg()
    is_outlier = asyncio.run(detect_outlier(query_text))
    print(f"Outlier detection result: {is_outlier}")
