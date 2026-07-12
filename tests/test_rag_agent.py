"""Unit tests for RAGAgent with mocked OpenAI dependencies."""

import json
from unittest.mock import MagicMock, patch

import pytest

from rag_agent import RAGAgent, flatten_json


# --- Tests for flatten_json ---


class TestFlattenJson:
    def test_flat_dict(self):
        result = flatten_json({"name": "Alice", "age": 30})
        assert "name: Alice" in result
        assert "age: 30" in result

    def test_nested_dict(self):
        result = flatten_json({"person": {"name": "Bob"}})
        assert "person:" in result
        assert "  name: Bob" in result

    def test_list(self):
        result = flatten_json(["a", "b", "c"])
        assert "[0]: a" in result
        assert "[1]: b" in result
        assert "[2]: c" in result

    def test_nested_list(self):
        result = flatten_json({"items": [1, 2, 3]})
        assert "items:" in result
        assert "  [0]: 1" in result

    def test_scalar(self):
        assert flatten_json("hello") == "hello"
        assert flatten_json(42) == "42"

    def test_empty_dict(self):
        assert flatten_json({}) == ""

    def test_empty_list(self):
        assert flatten_json([]) == ""

    def test_deeply_nested_hits_depth_limit(self):
        """Verify flatten_json doesn't crash on deeply nested structures."""
        # Build a structure 60 levels deep (exceeds _MAX_JSON_DEPTH=50)
        nested = "leaf"
        for _ in range(60):
            nested = {"level": nested}
        result = flatten_json(nested)
        assert "max depth exceeded" in result


# --- Fixtures ---


@pytest.fixture
def agent():
    """Create a RAGAgent with mocked OpenAI dependencies."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        with patch("rag_agent.OpenAIEmbeddings"), patch("rag_agent.ChatOpenAI"):
            return RAGAgent()


# --- Tests for RAGAgent initialization ---


class TestRAGAgentInit:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("rag_agent.OpenAIEmbeddings")
    @patch("rag_agent.ChatOpenAI")
    def test_init_success(self, mock_chat, mock_embeddings):
        agent = RAGAgent(model="gpt-4")
        assert agent.persist_directory == "./chroma_db"
        assert agent.vectorstore is None
        assert agent.retrieval_chain is None
        mock_chat.assert_called_once_with(model="gpt-4", temperature=0)
        mock_embeddings.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    def test_init_missing_api_key(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            RAGAgent()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("rag_agent.OpenAIEmbeddings")
    @patch("rag_agent.ChatOpenAI")
    def test_init_custom_persist_dir(self, mock_chat, mock_embeddings):
        agent = RAGAgent(persist_directory="/tmp/test_db")
        assert agent.persist_directory == "/tmp/test_db"


# --- Tests for load_documents ---


class TestLoadDocuments:
    def test_file_not_found(self, agent):
        with pytest.raises(FileNotFoundError, match="File not found"):
            agent.load_documents("/nonexistent/file.txt")

    def test_unsupported_file_type(self, agent, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            agent.load_documents(str(f))

    def test_file_too_large(self, agent, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")

        with patch("rag_agent.Path") as mock_path_cls:
            mock_path = MagicMock()
            mock_path_cls.return_value = mock_path
            mock_path.exists.return_value = True
            mock_path.stat.return_value = MagicMock(st_size=60 * 1024 * 1024)
            mock_path.suffix = ".txt"

            with pytest.raises(ValueError, match="File too large"):
                agent.load_documents(str(f))

    def test_load_txt_file(self, agent, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, this is test content.")
        docs = agent.load_documents(str(f))
        assert len(docs) >= 1
        assert "Hello" in docs[0].page_content

    def test_load_json_file(self, agent, tmp_path):
        f = tmp_path / "test.json"
        f.write_text(json.dumps({"key": "value", "nested": {"a": 1}}))
        docs = agent.load_documents(str(f))
        assert len(docs) == 1
        assert "key: value" in docs[0].page_content
        assert "nested:" in docs[0].page_content

    def test_load_invalid_json_file(self, agent, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{invalid json content")
        with pytest.raises(ValueError, match="Invalid JSON"):
            agent.load_documents(str(f))


# --- Tests for query validation ---


class TestQuery:
    def test_query_without_loading(self, agent):
        with pytest.raises(ValueError, match="No documents loaded"):
            agent.query("test question")

    def test_query_empty_string(self, agent):
        agent.retrieval_chain = MagicMock()
        with pytest.raises(ValueError, match="Question cannot be empty"):
            agent.query("")

    def test_query_whitespace_only(self, agent):
        agent.retrieval_chain = MagicMock()
        with pytest.raises(ValueError, match="Question cannot be empty"):
            agent.query("   ")

    def test_query_too_long(self, agent):
        agent.retrieval_chain = MagicMock()
        with pytest.raises(ValueError, match="Question too long"):
            agent.query("x" * 10001)

    def test_query_success(self, agent):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "input": "What is AI?",
            "answer": "AI is artificial intelligence.",
            "context": [],
        }
        agent.retrieval_chain = mock_chain

        result = agent.query("What is AI?")
        assert result == "AI is artificial intelligence."
        mock_chain.invoke.assert_called_once_with({"input": "What is AI?"})

    def test_query_with_sources(self, agent):
        mock_doc = MagicMock()
        mock_doc.page_content = "Source text"
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "input": "What is AI?",
            "answer": "AI is artificial intelligence.",
            "context": [mock_doc],
        }
        agent.retrieval_chain = mock_chain

        result = agent.query("What is AI?", return_sources=True)
        assert result["answer"] == "AI is artificial intelligence."
        assert len(result["sources"]) == 1


# --- Tests for ingest ---


class TestIngest:
    @patch("rag_agent.Chroma")
    def test_ingest_creates_vectorstore(self, mock_chroma_cls, agent, tmp_path):
        mock_vectorstore = MagicMock()
        mock_chroma_cls.from_documents.return_value = mock_vectorstore

        f = tmp_path / "test.txt"
        f.write_text("This is enough content to create at least one chunk for testing purposes.")

        agent.ingest(str(f))

        mock_chroma_cls.from_documents.assert_called_once()
        assert agent.vectorstore is mock_vectorstore
        assert agent.retrieval_chain is not None

    @patch("rag_agent.Chroma")
    def test_ingest_adds_to_existing_vectorstore(self, mock_chroma_cls, agent, tmp_path):
        mock_vectorstore = MagicMock()
        agent.vectorstore = mock_vectorstore

        f = tmp_path / "test.txt"
        f.write_text("More content for the existing vectorstore to process.")

        agent.ingest(str(f))

        mock_vectorstore.add_documents.assert_called_once()
        mock_chroma_cls.from_documents.assert_not_called()

    @patch("rag_agent.Chroma")
    def test_ingest_vectorstore_failure(self, mock_chroma_cls, agent, tmp_path):
        mock_chroma_cls.from_documents.side_effect = RuntimeError("DB write error")

        f = tmp_path / "test.txt"
        f.write_text("Content that will fail to persist to the vectorstore.")

        with pytest.raises(RuntimeError, match="Vectorstore write failed"):
            agent.ingest(str(f))

    def test_ingest_empty_content(self, agent, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("")

        with pytest.raises(ValueError, match="No content extracted"):
            agent.ingest(str(f))


# --- Tests for load_existing ---


class TestLoadExisting:
    def test_load_existing_no_directory(self, agent):
        agent.persist_directory = "/nonexistent/path"
        with pytest.raises(FileNotFoundError, match="No existing database"):
            agent.load_existing()

    @patch("rag_agent.Chroma")
    def test_load_existing_empty_vectorstore(self, mock_chroma_cls, agent, tmp_path):
        mock_vectorstore = MagicMock()
        mock_vectorstore.get.return_value = {"ids": []}
        mock_chroma_cls.return_value = mock_vectorstore

        agent.persist_directory = str(tmp_path)
        with pytest.raises(ValueError, match="contains no documents"):
            agent.load_existing()

    @patch("rag_agent.Chroma")
    def test_load_existing_success(self, mock_chroma_cls, agent, tmp_path):
        mock_vectorstore = MagicMock()
        mock_vectorstore.get.return_value = {"ids": ["doc1"]}
        mock_chroma_cls.return_value = mock_vectorstore

        agent.persist_directory = str(tmp_path)
        agent.load_existing()
        assert agent.vectorstore is mock_vectorstore
        assert agent.retrieval_chain is not None
