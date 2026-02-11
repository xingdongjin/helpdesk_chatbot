#!/usr/bin/env python3
"""
Web interface for the FluffyAI Helpdesk Chatbot using Gradio.
"""

import os
import sys

# Fix proxy URL before importing any libraries that might use it
for proxy_var in ['all_proxy', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    proxy_val = os.environ.get(proxy_var)
    if proxy_val and proxy_val.startswith('socks://'):
        os.environ[proxy_var] = proxy_val.replace('socks://', 'socks5://')

# Add src directory to Python path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
import gradio as gr

from src.chatbot import HelpdeskChatbot
from src.vector_store import VectorStore


def create_chatbot():
    """Initialize the chatbot with necessary components."""
    load_dotenv()

    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    print("Initializing vector store...")
    vector_store = VectorStore()

    doc_count = vector_store.get_collection_count()
    if doc_count == 0:
        raise ValueError("No documents found in vector store. Please run 'python src/ingest_data.py' first.")

    print(f"Loaded vector store with {doc_count} document chunks")

    chatbot = HelpdeskChatbot(openai_api_key, vector_store)
    return chatbot


def chat_interface(message, history, chatbot_instance):
    """Process user message and return response."""
    # Ensure message is a string
    message = str(message).strip() if message else ""

    if not message:
        return ""

    # Get response from chatbot
    response = chatbot_instance.chat(message)
    return response


def reset_conversation(chatbot_instance):
    """Reset the conversation history."""
    chatbot_instance.reset_conversation()
    return []


def create_ui():
    """Create and configure the Gradio interface."""

    # Initialize chatbot
    try:
        chatbot = create_chatbot()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        raise

    # Create Gradio interface
    with gr.Blocks(title="FluffyAI Helpdesk Chatbot") as demo:

        gr.Markdown(
            """
            # 🧸 FluffyAI Helpdesk Chatbot

            Welcome! I'm here to help with questions about our AI-powered plush toys.
            Ask me anything about our products, pricing, shipping, returns, or company policies!
            """
        )

        # Chat interface
        chatbot_ui = gr.Chatbot(
            label="Chat",
            height=500,
            show_label=False
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your question here...",
                show_label=False,
                scale=4,
                container=False
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("🔄 New Conversation", size="sm")

        gr.Markdown(
            """
            ---
            **About:** This chatbot uses RAG (Retrieval-Augmented Generation) with local embeddings
            to provide accurate information about FluffyAI products and policies.
            """
        )

        # Event handlers
        def user_submit(user_message, history):
            """Handle user message submission."""
            if history is None:
                history = []

            # Skip empty messages
            if not user_message or not str(user_message).strip():
                return "", history

            # Add user message in new Gradio 6.0 format
            history.append({"role": "user", "content": str(user_message).strip()})
            return "", history

        def bot_respond(history):
            """Generate bot response."""
            if not history or len(history) == 0:
                return history

            # Get last message
            last_message = history[-1]

            # Check if it's a user message
            if not isinstance(last_message, dict) or last_message.get("role") != "user":
                return history

            user_message = last_message.get("content", "")

            # Skip if empty
            if not user_message or not str(user_message).strip():
                return history

            # Get bot response
            bot_message = chat_interface(user_message, history, chatbot)

            # Add bot response in new Gradio 6.0 format
            history.append({"role": "assistant", "content": bot_message})
            return history

        def clear_chat():
            """Clear chat and reset conversation."""
            reset_conversation(chatbot)
            return []

        # Wire up events
        msg.submit(user_submit, [msg, chatbot_ui], [msg, chatbot_ui], queue=False).then(
            bot_respond, chatbot_ui, chatbot_ui
        )
        submit_btn.click(user_submit, [msg, chatbot_ui], [msg, chatbot_ui], queue=False).then(
            bot_respond, chatbot_ui, chatbot_ui
        )
        clear_btn.click(clear_chat, None, chatbot_ui, queue=False)

    return demo


if __name__ == "__main__":
    print("Starting FluffyAI Helpdesk Chatbot Web Interface...")

    try:
        demo = create_ui()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"\n❌ Failed to start web interface: {e}")
        print("\nMake sure you have:")
        print("  1. Created .env file with OPENAI_API_KEY")
        print("  2. Run 'python src/ingest_data.py' to load documents")
        print("  3. Installed gradio: pip install gradio")
        sys.exit(1)
