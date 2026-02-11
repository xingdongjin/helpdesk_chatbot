"""
Vector store using ChromaDB for document embeddings and retrieval.
Uses local sentence-transformers for embedding generation (no API costs!).
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import hashlib
import torch
import numpy as np


class VectorStore:
    """Manages document embeddings and similarity search using ChromaDB and local sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", collection_name: str = "helpdesk_docs"):
        """Initialize vector store with local sentence-transformers embeddings.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' which is fast on CPU (~90MB).
            collection_name: Name of the ChromaDB collection.
        """
        # Fix proxy URL if it uses 'socks://' instead of 'socks5://'
        for proxy_var in ['all_proxy', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            proxy_val = os.environ.get(proxy_var)
            if proxy_val and proxy_val.startswith('socks://'):
                os.environ[proxy_var] = proxy_val.replace('socks://', 'socks5://')

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Enable Intel GPU acceleration if available
        self.device = 'cpu'  # Default to CPU

        if torch.xpu.is_available():
            try:
                device_name = torch.xpu.get_device_name(0)
                self.model = self.model.to('xpu')
                self.device = 'xpu'
                print(f"✓ Model loaded on Intel GPU: {device_name}")
                print(f"  This will significantly speed up embedding generation!")
            except Exception as e:
                print(f"⚠ Failed to load model on XPU: {e}")
                print("  Falling back to CPU")
                self.device = 'cpu'
        else:
            print("✓ Model loaded on CPU")

        # Initialize ChromaDB (persistent storage)
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory="./chroma_db"
        ))

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")

    def _generate_id(self, text: str, source: str) -> str:
        """Generate unique ID for a document chunk."""
        content = f"{source}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 10):
        """Add documents to the vector store with embeddings."""
        texts = []
        metadatas = []
        ids = []

        for doc in documents:
            doc_id = self._generate_id(doc['content'], doc['source'])
            texts.append(doc['content'])
            metadatas.append({
                'source': doc['source'],
                'type': doc['type']
            })
            ids.append(doc_id)

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            # Generate embeddings using local model
            print(f"Generating embeddings for batch {i//batch_size + 1}... (device: {self.device})")
            embeddings = self.model.encode(batch_texts, show_progress_bar=False, device=self.device)

            # Convert to list (move from GPU to CPU if needed)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy().tolist()
            else:
                embeddings = embeddings.tolist()

            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

            print(f"Added {len(batch_texts)} documents to vector store")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents using semantic similarity."""
        # Generate query embedding using local model
        query_embedding = self.model.encode([query], show_progress_bar=False, device=self.device)[0]

        # Convert to list (move from GPU to CPU if needed)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy().tolist()
        else:
            query_embedding = query_embedding.tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        documents = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                documents.append({
                    'content': results['documents'][0][i],
                    'source': results['metadatas'][0][i]['source'],
                    'type': results['metadatas'][0][i]['type'],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })

        return documents

    def clear_collection(self):
        """Clear all documents from the collection."""
        self.chroma_client.delete_collection(name=self.collection.name)
        self.collection = self.chroma_client.create_collection(name=self.collection.name)
        print("Collection cleared")

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
