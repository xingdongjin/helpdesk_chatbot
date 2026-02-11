"""
Script to ingest all documents from the data directory into the vector store.
Run this once to set up the knowledge base, or whenever you update documents.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to Python path so imports work when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore


def main():
    """Ingest all documents from data directory."""
    print("=" * 50)
    print("FluffyAI Helpdesk - Document Ingestion")
    print("=" * 50)

    # Initialize components (no API key needed for local embeddings!)
    processor = DocumentProcessor()
    vector_store = VectorStore()

    # Clear existing data (optional - comment out to append instead)
    print("\nClearing existing vector store...")
    vector_store.clear_collection()

    # Load all documents from data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    print(f"\nLoading documents from: {data_dir}")

    documents = processor.load_directory(data_dir)
    print(f"Loaded {len(documents)} documents")

    # Load URLs from urls.txt if it exists
    urls_file = os.path.join(data_dir, 'business_info', 'urls.txt')
    if os.path.exists(urls_file):
        print(f"\nLoading URLs from: {urls_file}")
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

        for url in urls:
            print(f"  Fetching: {url}")
            doc = processor.load_webpage(url)
            if doc:
                documents.append(doc)
                print(f"    ✓ Loaded successfully")
            else:
                print(f"    ✗ Failed to load")

        print(f"Loaded {len(urls)} URLs")
    else:
        print(f"\nNo urls.txt found, skipping web page ingestion")

    print(f"\nTotal documents loaded: {len(documents)}")

    # Chunk documents for better retrieval
    chunked_docs = []
    for doc in documents:
        chunks = processor.chunk_text(doc['content'], chunk_size=800, overlap=150)
        for chunk in chunks:
            chunked_docs.append({
                'content': chunk,
                'source': doc['source'],
                'type': doc['type']
            })

    print(f"Created {len(chunked_docs)} chunks from documents")

    # Add to vector store
    print("\nGenerating embeddings locally (no API costs!)...")
    vector_store.add_documents(chunked_docs)

    print(f"\n✓ Successfully ingested documents!")
    print(f"Total chunks in database: {vector_store.get_collection_count()}")
    print("\nYou can now run the chatbot with: python src/main.py")


if __name__ == "__main__":
    main()
