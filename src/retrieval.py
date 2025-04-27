import os
import json
from typing import List, Dict
from embedding import embed_text 

def load_embeddings_from_file(embedding_file: str) -> List[Dict]:
    """Load embeddings and metadata from a JSON file."""
    with open(embedding_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def search(query: str, embeddings: List[Dict], top_k: int = 5) -> List[Dict]:
    """Search for the most similar chunks to the query."""
    query_embedding = embed_text(query)  # Assuming embed_text is defined in embedding.py
    results = []
    for item in embeddings:
        similarity = cosine_similarity(query_embedding, item['embedding'])
        results.append({
            "similarity": similarity,
            "metadata": item['metadata']
        })
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    embedding_file = 'data/embeddings.json'
    query = "Enter your search query here"

    embeddings = load_embeddings_from_file(embedding_file)
    results = search(query, embeddings)

    print(f"Top results for query: '{query}'")
    for result in results:
        print(f"Similarity: {result['similarity']:.4f}, Metadata: {result['metadata']}")