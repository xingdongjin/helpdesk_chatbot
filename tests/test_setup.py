#!/usr/bin/env python3
"""
Comprehensive setup validation script for FluffyAI Helpdesk Chatbot.
Run this after completing installation and data ingestion to verify everything works.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to Python path so imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_test(test_name):
    """Print test name."""
    print(f"\n[ ] {test_name}...", end=' ', flush=True)


def print_pass():
    """Print pass indicator."""
    print("\033[92m✓ PASS\033[0m")


def print_fail(message=""):
    """Print fail indicator with optional message."""
    print("\033[91m✗ FAIL\033[0m")
    if message:
        print(f"    Error: {message}")


def test_environment_file():
    """Test 1: Check if .env file exists."""
    print_test("Checking .env file exists")
    if os.path.exists('.env'):
        print_pass()
        return True
    else:
        print_fail(".env file not found. Run: cp .env.example .env")
        return False


def test_api_keys():
    """Test 2: Check if API keys are configured."""
    print_test("Validating API keys in .env")
    load_dotenv()

    openai_key = os.getenv('OPENAI_API_KEY')

    missing = []
    if not openai_key or openai_key == 'your_openai_api_key_here':
        missing.append("OPENAI_API_KEY")

    if missing:
        print_fail(f"Missing or invalid keys: {', '.join(missing)}")
        return False, None

    print_pass()
    return True, openai_key


def test_dependencies():
    """Test 3: Check if required packages are installed."""
    print_test("Checking required dependencies")
    required_packages = {
        'openai': 'openai',
        'chromadb': 'chromadb',
        'sentence-transformers': 'sentence_transformers',
        'python-dotenv': 'dotenv'
    }

    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print_fail(f"Missing packages: {', '.join(missing)}")
        print(f"    Run: pip install -r requirements.txt")
        return False

    print_pass()
    return True


def test_data_directory():
    """Test 4: Check if data directory exists and has files."""
    print_test("Checking data directory")

    if not os.path.exists('data'):
        print_fail("data/ directory not found")
        return False

    # Count files in data directory
    file_count = 0
    for root, dirs, files in os.walk('data'):
        file_count += len([f for f in files if not f.startswith('.')])

    if file_count == 0:
        print_fail("No data files found in data/ directory")
        return False

    print_pass()
    print(f"    Found {file_count} data files")
    return True


def test_vector_store():
    """Test 5: Check if vector store has data."""
    print_test("Checking vector store has ingested data")

    try:
        from src.vector_store import VectorStore

        vector_store = VectorStore()

        # Try to search for something
        results = vector_store.search("test query", top_k=1)

        if not results or len(results) == 0:
            print_fail("Vector store is empty. Run: ./python-xpu src/ingest_data.py")
            return False, None

        print_pass()
        print(f"    Vector store contains {vector_store.get_collection_count()} document chunks")
        return True, vector_store

    except Exception as e:
        print_fail(f"{str(e)}")
        return False, None


def test_retrieval(vector_store):
    """Test 6: Test vector store retrieval with sample queries."""
    print_test("Testing vector store retrieval")

    try:
        test_queries = [
            "What plush toys are available?",
            "How much does Buddy Bear cost?",
        ]

        for query in test_queries:
            results = vector_store.search(query, top_k=2)
            if not results or len(results) == 0:
                print_fail(f"No results for query: {query}")
                return False

        print_pass()
        return True

    except Exception as e:
        print_fail(f"{str(e)}")
        return False


def test_chatbot(openai_key, vector_store):
    """Test 7: Test chatbot initialization and simple query."""
    print_test("Testing chatbot initialization and response")

    try:
        from src.chatbot import HelpdeskChatbot

        chatbot = HelpdeskChatbot(openai_key, vector_store)

        # Test a simple query
        response = chatbot.chat("What toys do you have?")

        if not response or len(response) < 10:
            print_fail("Chatbot response too short or empty")
            return False

        print_pass()
        print(f"    Response preview: {response[:100]}...")
        return True

    except Exception as e:
        print_fail(f"{str(e)}")
        return False


def run_all_tests():
    """Run all validation tests."""
    print_header("FluffyAI Helpdesk Chatbot - Setup Validation")
    print("\nThis script will verify your installation and setup.\n")

    results = []

    # Test 1: Environment file
    results.append(test_environment_file())

    # Test 2: API keys
    api_test, openai_key = test_api_keys()
    results.append(api_test)

    if not api_test:
        print("\n\033[93m⚠ Cannot proceed without valid API keys.\033[0m")
        return False

    # Test 3: Dependencies
    results.append(test_dependencies())

    # Test 4: Data directory
    results.append(test_data_directory())

    # Test 5: Vector store
    vs_test, vector_store = test_vector_store()
    results.append(vs_test)

    if not vs_test:
        print("\n\033[93m⚠ Vector store not ready. Skipping chatbot tests.\033[0m")
        print_summary(results)
        return False

    # Test 6: Retrieval
    results.append(test_retrieval(vector_store))

    # Test 7: Chatbot
    results.append(test_chatbot(openai_key, vector_store))

    # Print summary
    print_summary(results)

    return all(results)


def print_summary(results):
    """Print test summary."""
    print_header("Test Summary")

    passed = sum(results)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")

    if all(results):
        print("\n\033[92m✓ ALL TESTS PASSED!\033[0m")
        print("\nYour helpdesk chatbot is ready to use!")
        print("\nNext steps:")
        print("  1. Run the chatbot: python src/main.py")
        print("  2. Run full test suite: python tests/test_queries.py")
    else:
        print("\n\033[91m✗ SOME TESTS FAILED\033[0m")
        print("\nPlease fix the issues above and run this script again.")
        print("\nQuick fixes:")
        print("  - Missing .env: cp .env.example .env")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - Empty vector store: python src/ingest_data.py")

    print()


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\033[91mUnexpected error: {str(e)}\033[0m")
        sys.exit(1)
