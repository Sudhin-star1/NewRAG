import pdfplumber
import nltk
import os
from typing import List, Dict

nltk.download('punkt')
nltk.download('punkt_tab')

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract text and metadata from PDF"""
    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages,1):
            text = page.extract_text()
            if text:
                pages.append({
                    "text": text,
                    "metadata": {
                        "filename": os.path.basename(pdf_path),
                        "page_number": page_num
                    }
                })
        return pages

def chunk_text(text:str, max_tokens: int = 200) -> List[str]:
    """Split text into chunks of approximately max_tokens."""
    sentences = nltk.sent_tokenize(text)

    chunks = [] # List of all chunks.
    current_chunk = [] # List of all sentences in the current chunk.
    current_length = 0 # Track number of token in current chunk.

    for sentence in sentences:
        sentence_tokens = len(nltk.word_tokenize(sentence))
        if current_length + sentence_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
            
    return chunks


def process_pdfs(input_dir:str, output_dir:str) -> List[Dict]:
    """Process all PDFs in the input directory and save the chunks to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_chunks = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            pages = extract_text_from_pdf(pdf_path)
            for page in pages:
                text = page['text']
                chunks = chunk_text(text)
                for chunk in chunks:
                    all_chunks.append({
                        "chunk": chunk,
                        "metadata": page['metadata']
                    })
    
    return all_chunks

def save_chunks_to_file(chunks: List[Dict], output_dir: str):
    """Save the chunks to a file in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, chunk in enumerate(chunks):
        filename = f"chunk_{i}.txt"
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            f.write(chunk['chunk'])
            f.write("\n\n")
            f.write(str(chunk['metadata']))
            f.write("\n\n")


if __name__ == "__main__":
    input_dir = 'data/raw'
    output_dir = 'data/processed'
    
    chunks = process_pdfs(input_dir, output_dir)
    save_chunks_to_file(chunks, output_dir)
    print(f"Processed {len(chunks)} chunks from PDFs in {input_dir} and saved to {output_dir}.")