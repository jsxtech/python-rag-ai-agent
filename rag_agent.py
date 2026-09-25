"""RAG AI Agent using LangChain and OpenAI."""

import json
import logging
import os
import warnings
from pathlib import Path

# Suppress langchain-community sunset warning — loaders don't have standalone packages yet
warnings.filterwarnings("ignore", message=".*langchain-community.*is being sunset.*")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 50
MAX_QUERY_LENGTH = 10000


_MAX_JSON_DEPTH = 50


def flatten_json(obj: object, prefix: str = "", _depth: int = 0) -> str:
    """Recursively flatten a JSON object into a human-readable string.

    Nested structures deeper than ``_MAX_JSON_DEPTH`` are truncated to avoid
    unbounded recursion; truncation is logged as a warning so ingestion is not
    silently lossy. Empty dicts/lists render inline (e.g. ``key: {}``) rather
    than emitting a dangling key followed by a blank line.
    """
    if _depth > _MAX_JSON_DEPTH:
        logger.warning(
            "JSON nesting exceeded max depth of %d at %r; content truncated",
            _MAX_JSON_DEPTH,
            prefix or "<root>",
        )
        return f"{prefix}... (max depth exceeded)"

    def _render(value: object, key_prefix: str, inline_label: str) -> None:
        """Append rendered lines for a single dict value or list item."""
        if isinstance(value, (dict, list)) and len(value) == 0:
            # Empty container: render inline to avoid a dangling key + blank line.
            marker = "{}" if isinstance(value, dict) else "[]"
            lines.append(f"{inline_label} {marker}")
        elif isinstance(value, (dict, list)):
            lines.append(inline_label)
            lines.append(flatten_json(value, key_prefix + "  ", _depth + 1))
        else:
            lines.append(f"{inline_label} {value}")

    lines: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            _render(value, prefix, f"{prefix}{key}:")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _render(item, prefix, f"{prefix}[{i}]:")
    else:
        return str(obj)
    return "\n".join(lines)


class RAGAgent:
    """A Retrieval-Augmented Generation agent for document Q&A."""

    def __init__(self, persist_directory: str = "./chroma_db", model: str = "gpt-3.5-turbo"):
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.retrieval_chain = None
        logger.info("RAGAgent initialized with model=%s", model)

    def load_documents(self, file_path: str) -> list[Document]:
        """Load documents from a file, selecting the appropriate loader by extension."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        max_size = MAX_FILE_SIZE_MB * 1024 * 1024
        file_size = path.stat().st_size
        if file_size > max_size:
            size_mb = file_size / 1024 / 1024
            raise ValueError(f"File too large: {size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")

        suffix = path.suffix.lower()
        loaders = {
            ".pdf": lambda: PyPDFLoader(file_path),
            ".txt": lambda: TextLoader(file_path, encoding="utf-8"),
            ".docx": lambda: UnstructuredWordDocumentLoader(file_path),
            ".doc": lambda: UnstructuredWordDocumentLoader(file_path),
            ".csv": lambda: CSVLoader(file_path, encoding="utf-8"),
            ".md": lambda: UnstructuredMarkdownLoader(file_path),
            ".markdown": lambda: UnstructuredMarkdownLoader(file_path),
            ".html": lambda: UnstructuredHTMLLoader(file_path),
            ".htm": lambda: UnstructuredHTMLLoader(file_path),
        }

        if suffix == ".json":
            return self._load_json(file_path)

        loader_factory = loaders.get(suffix)
        if loader_factory is None:
            raise ValueError(f"Unsupported file type: {suffix}")

        logger.debug("Loading %s file: %s", suffix, file_path)
        return loader_factory().load()

    def _load_json(self, file_path: str) -> list[Document]:
        """Load and flatten a JSON file into a Document."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}") from e

        content = flatten_json(data) if isinstance(data, (dict, list)) else str(data)
        if not content.strip():
            raise ValueError(f"JSON file contains no usable content: {file_path}")
        return [Document(page_content=content, metadata={"source": file_path})]

    def ingest(self, file_path: str) -> None:
        """Ingest a document into the vectorstore."""
        docs = self.load_documents(file_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError(f"No content extracted from {file_path}")

        logger.info("Ingesting %d chunks from %s", len(chunks), file_path)

        try:
            if self.vectorstore:
                self.vectorstore.add_documents(chunks)
            else:
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                )
        except Exception as e:
            logger.error("Failed to persist documents to vectorstore: %s", e)
            raise RuntimeError(f"Vectorstore write failed: {e}") from e

        self._setup_retrieval_chain()

    def load_existing(self) -> None:
        """Load an existing vectorstore from disk."""
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(f"No existing database at {self.persist_directory}")

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

        # Verify the vectorstore has documents using the public get() API
        result = self.vectorstore.get(limit=1)
        if not result.get("ids"):
            raise ValueError(
                f"Vectorstore at {self.persist_directory} exists but contains no documents"
            )

        logger.info("Loaded existing vectorstore from %s", self.persist_directory)
        self._setup_retrieval_chain()

    def _setup_retrieval_chain(self) -> None:
        """Build the retrieval chain using the current vectorstore."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Answer the user's question based on the following context. "
                        "If the context doesn't contain relevant information, say so.\n\n{context}"
                    ),
                ),
                ("human", "{input}"),
            ]
        )
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    def query(self, question: str, return_sources: bool = False) -> str | dict:
        """Query the RAG agent with a question.

        Args:
            question: The question to ask.
            return_sources: If True, return a dict with 'answer' and 'sources'.

        Returns:
            The answer string, or a dict with 'answer' and 'sources' if return_sources=True.
        """
        if not self.retrieval_chain:
            raise ValueError("No documents loaded. Call ingest() or load_existing() first.")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if len(question) > MAX_QUERY_LENGTH:
            raise ValueError(f"Question too long: {len(question)} chars (max {MAX_QUERY_LENGTH})")

        logger.debug("Querying: %s", question[:100])
        result = self.retrieval_chain.invoke({"input": question})

        if return_sources:
            return {
                "answer": result["answer"],
                "sources": result.get("context", []),
            }
        return result["answer"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    agent = RAGAgent()
    print("RAG Agent initialized. Use agent.ingest('file.pdf') to add documents.")
    print("Then use agent.query('your question') to ask questions.")
