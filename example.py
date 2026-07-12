"""Example usage of RAGAgent."""

import logging

from rag_agent import RAGAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Initialize agent
agent = RAGAgent()

# Ingest a document
agent.ingest("sample.txt")

# Simple query
answer = agent.query("What is this document about?")
print(f"Answer: {answer}\n")

# Query with sources
result = agent.query("What are the key topics?", return_sources=True)
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} chunks retrieved")
