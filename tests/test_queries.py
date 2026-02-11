"""
Test script with sample queries to evaluate the chatbot.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chatbot import HelpdeskChatbot
from vector_store import VectorStore


def run_test_queries():
    """Run a series of test queries and display results."""
    # Load environment variables
    load_dotenv()

    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        print("Error: OpenAI API key not found in .env file")
        return

    # Initialize chatbot
    print("Initializing chatbot...")
    vector_store = VectorStore()
    chatbot = HelpdeskChatbot(openai_api_key, vector_store)

    # Test queries
    test_queries = [
        "What AI plush toys do you have?",
        "How much does Buddy Bear cost?",
        "Do your toys work without internet?",
        "My toy broke, what should I do?",
        "Can I wash the plush toy?",
        "What's your return policy?",
        "Tell me about Dreamy Dragon",
        "Which toy is best for a 6-year-old who loves science?",
        "Do you ship internationally?",
        "How long does the battery last?",
        "Is my child's data safe?",
        "Can the toy speak Spanish?",
        "What's the difference between Buddy Bear and Robo Rabbit?",
        "Do you offer gift wrapping?",
        "What are your business hours?",
    ]

    print("\n" + "=" * 70)
    print("FluffyAI Helpdesk Chatbot - Test Queries")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}/{len(test_queries)}]")
        print(f"Query: {query}")
        print("-" * 70)

        response = chatbot.chat(query)
        print(f"Response: {response}")
        print("=" * 70)

        # Reset conversation between tests to avoid context mixing
        chatbot.reset_conversation()

    print("\n✓ All test queries completed!")


if __name__ == "__main__":
    run_test_queries()
