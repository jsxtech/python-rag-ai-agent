# Python RAG AI Agent

A minimal Retrieval-Augmented Generation (RAG) agent using LangChain and OpenAI.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` file with your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

```python
from rag_agent import RAGAgent

# Initialize (optional: custom model)
agent = RAGAgent(model="gpt-3.5-turbo")

# Ingest documents
agent.ingest("document.pdf")

# Query without sources
answer = agent.query("What is the main topic?")
print(answer)

# Query with sources
result = agent.query("What is the main topic?", return_sources=True)
print(result["answer"])
print(f"Sources: {len(result['sources'])} chunks")

# Load existing vectorstore
agent2 = RAGAgent()
agent2.load_existing()
answer = agent2.query("Follow-up question?")
```

## Testing

```bash
source .venv/bin/activate
pytest tests/ -v
```

Tests are fully mocked and do not require an OpenAI API key.

## Features

- Document ingestion (PDF, TXT, DOCX, CSV, Markdown, HTML, JSON)
- Vector storage with ChromaDB (persisted to `./chroma_db`)
- Semantic search retrieval (top 3 chunks)
- LLM-powered answer generation
- Input validation (empty queries, length limits)
- Structured logging via Python `logging` module
- File size limit: 50MB per document
- Chunk size: 1000 chars with 200 char overlap

## Supported File Types

- `.pdf` - PDF documents
- `.txt` - Plain text
- `.docx`, `.doc` - Word documents
- `.csv` - CSV files
- `.md`, `.markdown` - Markdown
- `.html`, `.htm` - HTML
- `.json` - JSON (auto-flattened)

## Limitations

- Requires OpenAI API key
- No document deletion/management
- Fixed chunk parameters
- UTF-8 encoding assumed for text files
