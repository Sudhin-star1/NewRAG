from flask import Flask, request, jsonify
from generation import generate_responses
import os

app = Flask(__name__)

# Define directories and files
INPUT_DIR = 'data/processed'
EMBEDDING_FILE = 'data/embeddings.json'

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload a PDF, process it, and return the processed chunks."""
    file = request.files.get('file')
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join('data/raw', file.filename)
        file.save(file_path)
        
        # Process PDF and save chunks
        from ingestion import process_pdfs, save_chunks_to_file
        output_dir = 'data/processed'
        chunks = process_pdfs('data/raw', output_dir)
        save_chunks_to_file(chunks, output_dir)
        
        return jsonify({"message": "PDF processed and chunks saved."}), 200
    return jsonify({"message": "Please upload a valid PDF."}), 400

@app.route('/query', methods=['GET'])
def query():
    """Query the system and get responses based on the processed PDF chunks."""
    query_text = request.args.get('query')
    if not query_text:
        return jsonify({"message": "Please provide a query parameter."}), 400

    # Generate responses based on the query
    responses = generate_responses(query_text, INPUT_DIR, EMBEDDING_FILE, top_k=5)
    
    return jsonify({
        "query": query_text,
        "results": [{
            "similarity": result['similarity'],
            "metadata": result['metadata']
        } for result in responses]
    }), 200

if __name__ == "__main__":
    app.run(debug=True)
