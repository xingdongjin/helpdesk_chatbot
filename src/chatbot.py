"""
Helpdesk chatbot with RAG capabilities using Kimi (Moonshot AI) and custom retrieval.
"""

import os
from openai import OpenAI
from typing import List, Dict, Optional

from .vector_store import VectorStore


class HelpdeskChatbot:
    """AI helpdesk chatbot with retrieval-augmented generation."""

    SYSTEM_PROMPT = """You are a helpful customer service agent for FluffyAI, a company that sells AI-powered plush toys.

Your personality:
- Professional yet friendly and approachable
- Concise - get to the point without being cold
- Slightly humorous when appropriate (gentle jokes, wordplay, but never at the customer's expense)
- Empathetic and understanding
- Patient with unclear questions

Your responsibilities:
- Answer questions about products, company policies, shipping, returns, etc.
- If a customer's question is unclear or ambiguous, politely ask for clarification
- If you don't know something or the information isn't in your knowledge base, be honest and offer to connect them with a human representative
- Stay on topic - focus on FluffyAI products and services
- Be helpful but don't make promises you can't keep

Guidelines:
- Keep responses under 150 words unless more detail is specifically requested
- Use the provided context to answer questions accurately
- If the context doesn't contain the answer, say so
- Never make up product features, prices, or policies
- When listing options, be clear and organized
- Add a touch of personality (e.g., "Great question!" or "I'd be happy to help with that!") but stay professional

Remember: You're here to help customers have a great experience with FluffyAI!"""

    def __init__(self, openai_api_key: str, vector_store: VectorStore, model: str = "moonshot-v1-8k"):
        """Initialize chatbot with Kimi (Moonshot AI) client and vector store."""
        # Fix proxy URL if it uses 'socks://' instead of 'socks5://'
        for proxy_var in ['all_proxy', 'ALL_PROXY', 'http_proxy', 'https_proxy']:
            proxy_val = os.environ.get(proxy_var)
            if proxy_val and proxy_val.startswith('socks://'):
                os.environ[proxy_var] = proxy_val.replace('socks://', 'socks5://')

        self.client = OpenAI(api_key=openai_api_key, base_url="https://api.moonshot.cn/v1")
        self.vector_store = vector_store
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []

    def _retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context from vector store."""
        results = self.vector_store.search(query, top_k=top_k)

        if not results:
            return "No relevant information found in knowledge base."

        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"[Source {i}: {doc['source']}]\n{doc['content']}")

        return "\n\n---\n\n".join(context_parts)

    def chat(self, user_message: str, use_rag: bool = True) -> str:
        """Process user message and generate response."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Prepare messages for API call
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Add context if RAG is enabled
        if use_rag:
            context = self._retrieve_context(user_message)
            context_message = f"""Here is relevant information from the knowledge base that may help answer the user's question:

{context}

---

Now, please answer the user's question based on this context. If the context doesn't contain the answer, let the user know and offer to help in another way."""

            messages.append({
                "role": "system",
                "content": context_message
            })

        # Add conversation history (last 10 messages to keep context window manageable)
        messages.extend(self.conversation_history[-10:])

        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )

        assistant_message = response.choices[0].message.content

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history."""
        return self.conversation_history
