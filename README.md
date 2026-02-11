# FluffyAI Helpdesk Chatbot

An intelligent customer service chatbot for FluffyAI's AI-powered plush toys business. Built with RAG (Retrieval-Augmented Generation) using Kimi (Moonshot AI) and local sentence-transformers embeddings.

## Quick Start

```bash
# 1. Create a Python virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env and add your Moonshot API key

# 5. Ingest documents
python src/ingest_data.py

# 6. Run web interface
python app.py
# Open browser to http://localhost:7860
```

## Features

- **Web & CLI Interfaces**: Modern web UI with Gradio, plus command-line option
- **Multi-format Document Support**: Ingests Markdown, PDF, plain text, and web pages
- **RAG Architecture**: Retrieval-augmented generation for accurate, context-aware responses
- **Professional Personality**: Concise, friendly, and slightly humorous tone
- **Local Embeddings**: No embedding API costs - uses sentence-transformers locally
- **Intel GPU Support**: Optional Intel XPU acceleration for faster embeddings
- **CPU-Optimized**: Runs efficiently on CPU if GPU is not available
- **Conversation Memory**: Maintains context throughout the conversation
- **Clarification Handling**: Asks for clarification when user intent is unclear

## Architecture

- **Document Processing**: Loads and chunks documents from various sources
- **Vector Store**: ChromaDB with local sentence-transformers embeddings for semantic search
- **Embeddings**: all-MiniLM-L6-v2 model running locally (CPU or Intel GPU)
- **LLM**: Kimi (Moonshot AI) moonshot-v1-8k model for natural language generation
- **RAG Pipeline**: Retrieves relevant context before generating responses

## Prerequisites

- Python 3.8+
- Moonshot AI API key (get from https://platform.moonshot.cn/)
- (Optional) Intel GPU with XPU support for faster embedding generation

## Installation

1. Clone or navigate to the project directory:
```bash
cd helpdesk_chatbot_new
```

2. Create and activate a Python virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```

5. Edit `.env` and add your Moonshot API key:
```
OPENAI_API_KEY=your_moonshot_api_key_here
```

Note: Get your API key from https://platform.moonshot.cn/. The key is stored in `OPENAI_API_KEY` for compatibility with the OpenAI SDK. Embeddings run locally using sentence-transformers.

6. Verify setup (optional):
```bash
python test_setup.py
```

## Usage

### Step 1: Ingest Documents

First, load your business and product information into the vector database:

```bash
python src/ingest_data.py
```

This will:
- Load all documents from the `data/` directory
- Split them into chunks for optimal retrieval
- Generate embeddings using local sentence-transformers model
- Store them in ChromaDB

The first run will download the sentence-transformers model (~90MB), which is then cached locally.

### Step 2: Run the Chatbot

You can run the chatbot in two ways:

#### Option A: Web Interface (Recommended)

Start the web interface with Gradio:

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

Features:
- Clean, modern chat interface
- Easy conversation management
- Works on any device with a browser
- No command-line needed

#### Option B: Command-Line Interface

Start the interactive CLI chatbot:

```bash
python src/main.py
```

Available commands during conversation:
- Type your question normally to chat
- `reset` - Start a new conversation
- `quit` or `exit` - End the session

### Step 3: Run Test Queries (Optional)

Test the chatbot with predefined queries:

```bash
python tests/test_queries.py
```

## Intel GPU Acceleration (Optional)

The project supports Intel GPU (XPU) acceleration for faster embedding generation on Intel Arc and Iris Xe graphics. This works with both the web and CLI versions.

### Using Intel GPU

If you have an Intel GPU and want to use hardware acceleration:

```bash
# Run web interface with Intel GPU support
./python-xpu app.py

# Or run CLI with Intel GPU support
./python-xpu src/main.py

# Or run data ingestion with Intel GPU support
./python-xpu src/ingest_data.py
```

The chatbot will automatically detect and use Intel GPU if available. You'll see:
- `✓ Model loaded on Intel GPU: [GPU name]` - GPU acceleration active
- `✓ Model loaded on CPU` - Running on CPU (fallback)

### Requirements for Intel GPU

- Intel Arc or Iris Xe integrated graphics
- PyTorch with XPU support (2.8.0+)
- Intel GPU drivers and compute runtime

No code changes needed - the system automatically falls back to CPU if GPU is not available.

## Project Structure

```
helpdesk_chatbot_new/
├── data/                          # Knowledge base documents
│   ├── business_info/            # Company info, FAQ, policies
│   │   ├── company_overview.md
│   │   └── faq.txt
│   └── products/                 # Product descriptions
│       ├── buddy_bear.md
│       ├── robo_rabbit.md
│       └── dreamy_dragon.md
├── src/                          # Source code
│   ├── document_processor.py    # Document loading and chunking
│   ├── vector_store.py          # Vector DB and embeddings
│   ├── chatbot.py               # Main chatbot logic
│   ├── ingest_data.py           # Data ingestion script
│   └── main.py                  # Interactive CLI
├── tests/                        # Test scripts
│   └── test_queries.py
├── chroma_db/                    # ChromaDB storage (auto-created)
├── app.py                        # Web interface (Gradio)
├── requirements.txt              # Python dependencies
├── test_setup.py                 # Setup validation script
├── test_gpu.py                   # GPU detection and performance test
├── python-xpu                    # Intel GPU wrapper script (optional)
├── .env.example                  # Environment variables template
├── .env                          # Environment variables (create this)
└── README.md                     # This file
```

## Adding Your Own Documents

To add your own business and product information:

1. **Markdown files** (`.md`): Add to `data/business_info/` or `data/products/`
2. **PDF files** (`.pdf`): Place in any subdirectory under `data/`
3. **Text files** (`.txt`): Place in any subdirectory under `data/`
4. **Web pages**: Add URLs (one per line) to `data/business_info/urls.txt`
   - Lines starting with `#` are treated as comments
   - Example:
     ```
     # Company websites
     https://example.com
     https://example.com/faq
     ```

After adding documents, re-run the ingestion to update embeddings:
```bash
python src/ingest_data.py
```

Note: Embedding generation runs locally, so no API costs for adding new documents!

## Customization

### Change Embedding Model

Edit the model name in `src/vector_store.py`:
```python
vector_store = VectorStore(model_name="all-mpnet-base-v2")  # Larger, more accurate
# Default is "all-MiniLM-L6-v2" (fast, 90MB)
```

### Adjust Chatbot Personality

Edit the `SYSTEM_PROMPT` in `src/chatbot.py` to change tone and behavior.

### Change Chunk Size

Modify chunk parameters in `src/ingest_data.py`:
```python
chunks = processor.chunk_text(doc['content'], chunk_size=800, overlap=150)
```

### Change LLM Model

Pass a different Kimi model when initializing the chatbot:
```python
# Available models: moonshot-v1-8k (default), moonshot-v1-32k, moonshot-v1-128k
chatbot = HelpdeskChatbot(openai_api_key, vector_store, model="moonshot-v1-32k")
```

### Adjust Retrieval

Change the number of retrieved documents in `src/chatbot.py`:
```python
results = self.vector_store.search(query, top_k=5)  # Default is 3
```

## Sample Data

The project includes sample data for a fictional company "FluffyAI" that sells AI-powered plush toys:

- **Company Overview**: Business hours, contact info, shipping, returns
- **FAQ**: Common customer questions
- **Products**:
  - Buddy Bear ($79.99) - Classic companion for ages 3-10
  - Robo Rabbit ($89.99) - STEM-focused for ages 5-12
  - Dreamy Dragon ($94.99) - Mindfulness and creativity for ages 4-12

## Performance Notes

- **Local Embeddings**: No API calls for embeddings - runs entirely locally
- **No Embedding Costs**: Sentence-transformers eliminates embedding API costs
- **GPU Acceleration**: Intel XPU support speeds up embedding generation significantly
- **CPU Fallback**: Works efficiently on CPU if GPU is not available
- **Efficient Retrieval**: ChromaDB provides fast similarity search
- **Token Optimization**: Chunks are sized to balance context and cost
- **API Costs**: Uses `moonshot-v1-8k` by default for cost-efficient responses
- **First Run**: Downloads embedding model (~90MB) on first use, then cached locally

## Troubleshooting

**"No documents found in vector store"**
- Run `python src/ingest_data.py` first

**"No module named 'vector_store'"**
- Make sure you're running scripts from the project root directory
- The test_setup.py script handles imports correctly

**API Key Errors**
- Check that `.env` file exists (copy from `.env.example` if needed)
- Ensure it contains valid OPENAI_API_KEY with your Moonshot API key
- Get your API key from https://platform.moonshot.cn/

**Import Errors**
- Verify all dependencies are installed: `pip install -r requirements.txt`
- First run will download the sentence-transformers model

**Slow Embedding Generation**
- Consider using Intel GPU acceleration with `./python-xpu` wrapper
- Or use a smaller embedding model in vector_store.py

**Slow Chat Responses**
- Check your internet connection (requires Moonshot AI API calls)
- The default `moonshot-v1-8k` model is optimized for speed
- For longer context, use `moonshot-v1-32k` or `moonshot-v1-128k`

**GPU Not Detected**
- Verify Intel GPU drivers and compute runtime are installed
- Check with: `python test_gpu.py`
- The system will automatically fall back to CPU

## Future Enhancements

- ✅ Web interface with Gradio (implemented!)
- Multi-language support
- Conversation export/import
- Analytics dashboard
- Integration with ticketing systems
- Voice interface
- Support for additional embedding models
- Batch processing for large document sets
- User authentication for web interface
- Conversation history persistence

## License

This is a sample project for educational purposes.
