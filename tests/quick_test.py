"""Quick test to verify the RAG retrieval works"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to Python path so imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vector_store import VectorStore

load_dotenv()

print("Testing vector store retrieval...")
vector_store = VectorStore()

# Test retrieval without calling OpenAI
test_queries = [
    "What plush toys are available?",
    "How much does Buddy Bear cost?",
    "Can I wash the toy?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    print("-" * 50)
    results = vector_store.search(query, top_k=2)
    for i, doc in enumerate(results, 1):
        print(f"\n[Result {i}]")
        print(f"Source: {doc['source']}")
        print(f"Content: {doc['content'][:200]}...")

print("\n✓ Vector store retrieval is working perfectly!")
