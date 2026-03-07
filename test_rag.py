from rag_agent import RAGAgent

def test_rag_agent():
    print("=== Testing RAG Agent ===\n")
    
    # Test 1: Initialize agent
    print("1. Initializing agent...")
    agent = RAGAgent()
    print("✓ Agent initialized\n")
    
    # Test 2: Ingest document
    print("2. Ingesting sample.txt...")
    agent.ingest("sample.txt")
    print("✓ Document ingested\n")
    
    # Test 3: Query without sources
    print("3. Testing query without sources...")
    answer = agent.query("What is Machine Learning?")
    print(f"Q: What is Machine Learning?")
    print(f"A: {answer}\n")
    
    # Test 4: Query with sources
    print("4. Testing query with sources...")
    result = agent.query("What are the key applications of AI?", return_sources=True)
    print(f"Q: What are the key applications of AI?")
    print(f"A: {result['answer']}")
    print(f"Sources: {len(result['sources'])} chunks retrieved\n")
    
    # Test 5: Multiple document ingestion
    print("5. Testing persistence...")
    agent2 = RAGAgent()
    agent2.load_existing()
    answer2 = agent2.query("What is Deep Learning?")
    print(f"Q: What is Deep Learning?")
    print(f"A: {answer2}\n")
    
    print("=== All tests passed! ===")

if __name__ == "__main__":
    test_rag_agent()
