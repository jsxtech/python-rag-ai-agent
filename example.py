from rag_agent import RAGAgent

# Initialize agent
agent = RAGAgent()

# Ingest a document
agent.ingest("sample.txt")

# Query the agent
response = agent.query("What is this document about?")
print(response)
