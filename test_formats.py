from rag_agent import RAGAgent

def test_multiple_formats():
    print("=== Testing Multiple File Formats ===\n")
    
    agent = RAGAgent()
    
    # Test different file types
    files = [
        ("sample.txt", "TXT"),
        ("sample.json", "JSON"),
        ("sample.md", "Markdown"),
        ("sample.csv", "CSV")
    ]
    
    for file, file_type in files:
        print(f"Ingesting {file_type} file: {file}")
        try:
            agent.ingest(file)
            print(f"✓ {file_type} file ingested successfully\n")
        except Exception as e:
            print(f"✗ Error: {e}\n")
    
    # Test queries
    queries = [
        "What is Machine Learning?",
        "Who works at TechCorp?",
        "What is markdown used for?",
        "Who lives in Seattle?"
    ]
    
    print("=== Testing Queries ===\n")
    for q in queries:
        print(f"Q: {q}")
        answer = agent.query(q)
        print(f"A: {answer}\n")
    
    print("=== All tests completed! ===")

if __name__ == "__main__":
    test_multiple_formats()
