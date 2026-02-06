"""Interactive RAG System using MultiRAGEngine"""

from multi_rag_engine import MultiRAGEngine


def main():
    """Run interactive RAG system"""
    try:
        # Initialize Multi RAG Engine
        rag = MultiRAGEngine()
        
        # Estado actual
        current_db = "cliente"
        
        # Interactive Query Loop
        print("\n" + "="*60)
        print("RAG System Ready!")
        print("Commands:")
        print("  'q'                 -> quit")
        print("  'update'            -> refresh ALL databases")
        print("  'use cliente'       -> switch to client DB (only datasetGen)")
        print("  'use distribuidor'  -> switch to distributor DB (dataSet + datasetGen)")
        print("="*60 + "\n")
        
        while True:
            prompt = f"\n[{current_db}] Your question: "
            query_text = input(prompt).strip()
            
            if query_text.lower() == 'q':
                print("Goodbye!")
                break
            
            if query_text.lower() == 'update':
                rag.update_database()
                continue
                
            if query_text.lower() == 'use cliente':
                current_db = "cliente"
                print("Switched to CLIENTE database (Restricted Access)")
                continue
                
            if query_text.lower() == 'use distribuidor':
                current_db = "distribuidor"
                print("Switched to DISTRIBUIDOR database (Full Access)")
                continue
            
            if not query_text:
                print("Please enter a question.")
                continue
            
            print(f"\nProcessing in {current_db}...")
            result = rag.ask(query_text, db_type=current_db)
            print("\n--- Answer ---")
            print(result)
            print("--------------")
            
    except Exception as e:
        print(f"Error initializing RAG Engine: {e}")


if __name__ == "__main__":
    main()
