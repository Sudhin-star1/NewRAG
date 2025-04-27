# generation.py

import os
import json
from typing import List, Dict
from retrieval import search, load_embeddings_from_file
from embedding import load_chunks_from_file, process_embeddings, save_embeddings_to_file

def ensure_embeddings(input_dir: str, embedding_file: str) -> List[Dict]:
    """Ensure embeddings exist; if not, generate them."""
    if not os.path.exists(embedding_file):
        print("Embeddings not found. Generating embeddings...")
        chunks = load_chunks_from_file(input_dir)
        embedded_chunks = process_embeddings(chunks)
        save_embeddings_to_file(embedded_chunks, embedding_file)
        return embedded_chunks
    else:
        return load_embeddings_from_file(embedding_file)

def generate_responses(query: str, input_dir: str, embedding_file: str, top_k: int = 5) -> List[Dict]:
    """
    Generate responses for a given query by searching through preprocessed text chunks.
    """
    # Step 1: Load or generate embeddings
    embedded_chunks = ensure_embeddings(input_dir, embedding_file)

    # Step 2: Perform search
    results = search(query, embedded_chunks, top_k=top_k)
    return results

if __name__ == "__main__":
    input_dir = 'data/processed'
    embedding_file = 'data/embeddings.json'
    query = input("Enter your search query: ")
    top_k = 5

    responses = generate_responses(query, input_dir, embedding_file, top_k)
    
    print(f"\nTop {top_k} results for query: '{query}'\n")
    for idx, response in enumerate(responses, 1):
        print(f"{idx}. Similarity: {response['similarity']:.4f}, Metadata: {response['metadata']}")
