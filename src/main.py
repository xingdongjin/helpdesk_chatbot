"""
Main script to run the helpdesk chatbot in interactive mode.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to Python path so imports work when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.chatbot import HelpdeskChatbot
from src.vector_store import VectorStore


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 60)
    print("🧸 FluffyAI Helpdesk Chatbot")
    print("=" * 60)
    print("Welcome! I'm here to help with questions about our AI plush toys.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("Type 'reset' to start a new conversation.")
    print("=" * 60 + "\n")


def main():
    """Run the interactive chatbot."""
    # Load environment variables
    load_dotenv()

    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        print("Error: OpenAI API key not found!")
        print("Please set OPENAI_API_KEY in your .env file")
        return

    # Initialize components
    print("Initializing chatbot...")
    vector_store = VectorStore()

    # Check if documents are loaded
    doc_count = vector_store.get_collection_count()
    if doc_count == 0:
        print("\n⚠️  Warning: No documents found in vector store!")
        print("Please run 'python src/ingest_data.py' first to load documents.")
        return

    print(f"Loaded vector store with {doc_count} document chunks")

    chatbot = HelpdeskChatbot(openai_api_key, vector_store)

    print_header()

    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\nChatbot: Thanks for chatting! Have a fluffy day! 🧸")
                break

            # Check for reset command
            if user_input.lower() == 'reset':
                chatbot.reset_conversation()
                print("\n✓ Conversation reset. Starting fresh!\n")
                continue

            # Generate response
            response = chatbot.chat(user_input)
            print(f"\nChatbot: {response}\n")

        except KeyboardInterrupt:
            print("\n\nChatbot: Thanks for chatting! Have a fluffy day! 🧸")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    main()
