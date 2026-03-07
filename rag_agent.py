import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)
from langchain.schema import Document
from langchain.chains import RetrievalQA

load_dotenv()

class RAGAgent:
    def __init__(self, persist_directory="./chroma_db", model="gpt-3.5-turbo"):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self, file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size (50MB limit)
        max_size = 50 * 1024 * 1024
        if path.stat().st_size > max_size:
            size_mb = path.stat().st_size / 1024 / 1024
            raise ValueError(f"File too large: {size_mb:.1f}MB (max 50MB)")
        
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            loader = PyPDFLoader(file_path)
        elif suffix == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif suffix in ['.docx', '.doc']:
            loader = UnstructuredWordDocumentLoader(file_path)
        elif suffix == '.csv':
            loader = CSVLoader(file_path, encoding='utf-8')
        elif suffix in ['.md', '.markdown']:
            loader = UnstructuredMarkdownLoader(file_path)
        elif suffix in ['.html', '.htm']:
            loader = UnstructuredHTMLLoader(file_path)
        elif suffix == '.json':
            return self._load_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        return loader.load()
    
    def _load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def flatten_json(obj, prefix=''):
            lines = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        lines.append(f"{prefix}{key}:")
                        lines.append(flatten_json(value, prefix + '  '))
                    else:
                        lines.append(f"{prefix}{key}: {value}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        lines.append(f"{prefix}[{i}]:")
                        lines.append(flatten_json(item, prefix + '  '))
                    else:
                        lines.append(f"{prefix}[{i}]: {item}")
            else:
                return str(obj)
            return '\n'.join(lines)
        
        content = flatten_json(data) if isinstance(data, (dict, list)) else str(data)
        return [Document(page_content=content, metadata={"source": file_path})]
    
    def ingest(self, file_path):
        docs = self.load_documents(file_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        if not chunks:
            raise ValueError(f"No content extracted from {file_path}")
        
        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        self._setup_qa_chain()
        
    def load_existing(self):
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(f"No existing database at {self.persist_directory}")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self._setup_qa_chain()
        
    def _setup_qa_chain(self):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
    
    def query(self, question, return_sources=False):
        if not self.qa_chain:
            raise ValueError("No documents loaded. Call ingest() or load_existing() first.")
        result = self.qa_chain.invoke({"query": question})
        if return_sources:
            return {"answer": result["result"], "sources": result["source_documents"]}
        return result["result"]

if __name__ == "__main__":
    agent = RAGAgent()
    
    # Example usage
    print("RAG Agent initialized. Use agent.ingest('file.pdf') to add documents.")
    print("Then use agent.query('your question') to ask questions.")
