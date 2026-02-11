"""
Document processor for ingesting various file formats into the vector database.
Supports: Markdown, PDF, TXT, and web pages.
"""

import os
from typing import List, Dict
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import markdown


class DocumentProcessor:
    """Handles loading and processing of various document types."""

    def __init__(self):
        self.documents = []

    def load_markdown(self, file_path: str) -> Dict[str, str]:
        """Load and parse markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Convert to plain text (removing markdown syntax for better embedding)
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()

        return {
            'content': text,
            'source': file_path,
            'type': 'markdown'
        }

    def load_pdf(self, file_path: str) -> Dict[str, str]:
        """Load and extract text from PDF."""
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() + "\n"

        return {
            'content': text.strip(),
            'source': file_path,
            'type': 'pdf'
        }

    def load_text(self, file_path: str) -> Dict[str, str]:
        """Load plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            'content': content,
            'source': file_path,
            'type': 'text'
        }

    def load_webpage(self, url: str) -> Dict[str, str]:
        """Fetch and extract text from webpage."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()

            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return {
                'content': text,
                'source': url,
                'type': 'webpage'
            }
        except Exception as e:
            print(f"Error loading webpage {url}: {e}")
            return None

    def load_directory(self, directory_path: str) -> List[Dict[str, str]]:
        """Recursively load all supported files from a directory."""
        documents = []
        directory = Path(directory_path)

        for file_path in directory.rglob('*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()

                try:
                    if suffix == '.md':
                        doc = self.load_markdown(str(file_path))
                        documents.append(doc)
                    elif suffix == '.pdf':
                        doc = self.load_pdf(str(file_path))
                        documents.append(doc)
                    elif suffix == '.txt':
                        doc = self.load_text(str(file_path))
                        documents.append(doc)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        return documents

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better retrieval."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks
