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
- `.docx`, `.doc` - Word documents (see note below)
- `.csv` - CSV files
- `.md`, `.markdown` - Markdown
- `.html`, `.htm` - HTML
- `.json` - JSON (auto-flattened)

## Limitations

- Requires OpenAI API key
- No document deletion/management
- Fixed chunk parameters
- UTF-8 encoding assumed for text files

## Notes

- Legacy `.doc` (binary Word) files are loaded via `unstructured`, which relies on
  a system binary (`libreoffice` or `antiword`) that is **not** installed by pip.
  If `.doc` ingestion fails, install one of those tools or convert the file to
  `.docx` first. Modern `.docx` files work out of the box.

## Security

Run `pip-audit` (available via `requirements-dev.txt`) to check dependencies for
known vulnerabilities.

- `pypdf` is pinned to `>=6.16.1` to pick up fixes for a batch of 2026 parsing
  advisories (malicious-PDF DoS). This matters because `pypdf` parses untrusted
  input in the ingestion path.
- `chromadb 1.5.9` (the latest release as of this writing) has open advisories
  (PYSEC-2026-311, -3813, -3814, -3815) with **no patched version available yet**.
  These cannot be resolved by upgrading; monitor for a fixed release and bump the
  pin when one ships.
