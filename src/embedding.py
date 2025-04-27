import os
import json
from typing import List, Dict

def load_chunks_from_file(input_dir):
    """Load chunks and their metadata from text files."""
    chunks = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines if line.strip() != '']

                if len(lines) < 2:
                    raise ValueError(f"File {filename} doesn't have enough lines.")

                # The last line should be metadata
                metadata_line = lines[-1]
                try:
                    metadata = json.loads(metadata_line)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid metadata JSON in {filename}: {metadata_line}")

                # All previous lines are the chunk text
                chunk_text = " ".join(lines[:-1])

                chunks.append({
                    'chunk': chunk_text,
                    'metadata': metadata
                })

    return chunks

def embed_text(text: str) -> List[float]:
    """Generate an embedding for the given text using a hypothetical embedding model."""
    # Replace this with actual embedding logic, e.g., using OpenAI, Hugging Face, etc.
    return [0.0] * 512  # Placeholder for a 512-dimensional embedding

def process_embeddings(chunks: List[Dict]) -> List[Dict]:
    """Generate embeddings for all text chunks."""
    embedded_chunks = []
    for chunk in chunks:
        embedding = embed_text(chunk['chunk'])
        embedded_chunks.append({
            "embedding": embedding,
            "metadata": chunk['metadata']
        })
    return embedded_chunks

def save_embeddings_to_file(embedded_chunks: List[Dict], output_file: str):
    """Save the embeddings and metadata to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embedded_chunks, f, indent=4)

if __name__ == "__main__":
    input_dir = 'data/processed'
    output_file = 'data/embeddings.json'

    chunks = load_chunks_from_file(input_dir)
    embedded_chunks = process_embeddings(chunks)
    save_embeddings_to_file(embedded_chunks, output_file)
    print(f"Generated embeddings for {len(embedded_chunks)} chunks and saved to {output_file}.")