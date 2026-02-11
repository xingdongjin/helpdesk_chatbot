# Quick Start Guide

Get your FluffyAI helpdesk chatbot running in 3 simple steps!

## Step 1: Install Dependencies

```bash
cd helpdesk_chatbot
pip install -r requirements.txt
```

## Step 2: Set Up API Key

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Get your Moonshot AI API key:
   - Visit https://platform.moonshot.cn/
   - Sign up or log in
   - Generate an API key

3. Edit `.env` and add your API key:
```bash
# Open with your favorite editor
nano .env
# OR
vim .env
# OR
code .env
```

Add your key:
```
OPENAI_API_KEY=your_moonshot_api_key_here
```

Note: The variable is named `OPENAI_API_KEY` for compatibility with the OpenAI SDK, but you should use your Moonshot AI key here.

## Step 3: Ingest Sample Data

Load the sample FluffyAI business data:
```bash
python src/ingest_data.py
```

You should see output like:
```
Loading documents from: ../data
Loaded 5 documents
Created 23 chunks from documents
Generating embeddings and storing in vector database...
✓ Successfully ingested documents!
```

## Step 4: Run the Chatbot

### Option A: Web Interface (Recommended)

Start the web interface:
```bash
python src/app.py
```

Then open your browser to `http://localhost:7860`

### Option B: Command-Line Interface

Start chatting in the terminal:
```bash
python src/main.py
```

Try these example questions:
- "What AI plush toys do you have?"
- "How much does Buddy Bear cost?"
- "Can I wash the toy?"
- "What's your return policy?"
- "Which toy is best for learning science?"

## Optional: Run Test Suite

Test all functionality:
```bash
python tests/test_queries.py
```

## Troubleshooting

**Can't install chromadb?**
- Make sure you have Python 3.8 or higher
- Try: `pip install --upgrade pip`

**API key errors?**
- Get your API key from https://platform.moonshot.cn/
- Verify the key is correct in `.env` file
- Check that there are no extra spaces or quotes
- Make sure the key is assigned to `OPENAI_API_KEY`

**No documents found?**
- Make sure you ran `python src/ingest_data.py`
- Check that files exist in `data/business_info/` and `data/products/`

## Next Steps

- Replace sample data with your own business information
- Customize the chatbot personality in `src/chatbot.py`
- Try different Kimi models (moonshot-v1-32k, moonshot-v1-128k)
- Deploy to a server for customer access

Happy chatting!
