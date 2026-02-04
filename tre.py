"""Interactive RAG System using RAGEngine"""

from rag_engine import RAGEngine


def main():
    """Run interactive RAG system"""
    try:
        # Initialize RAG Engine
        rag = RAGEngine(data_dir="dataSet")
        
        # Interactive Query Loop
        print("\n" + "="*60)
        print("RAG System Ready! Type 'q' to quit, 'update' to refresh DB.")
        print("="*60 + "\n")
        
        while True:
            query_text = input("\nYour question: ").strip()
            
            if query_text.lower() == 'q':
                print("Goodbye!")
                break
            
            if query_text.lower() == 'update':
                rag.update_database()
                continue
            
            if not query_text:
                print("Please enter a question.")
                continue
            
            print(f"\nProcessing...")
            result = rag.ask(query_text)
            print("\n--- Answer ---")
            print(result)
            print("--------------")
            
    except Exception as e:
        print(f"Error initializing RAG Engine: {e}")


if __name__ == "__main__":
    main()
