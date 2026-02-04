import os
import sys
import warnings
import glob



from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


# Configuration
DATA_DIR = "dataSet"  # Directorio donde están los archivos
FILE_EXTENSIONS = ["*.txt", "*.md"]  # Extensiones a procesar
EMBEDDING_MODEL_NAME = "qwen3-embedding:8b"
LLM_MODEL_NAME = "gemma3:4b"
COLLECTION_NAME = "local-rag"
PERSIST_DIRECTORY = "./chroma_db"  # Directorio de persistencia

def load_documents():
    """Load all .txt and .md files from the data directory"""
    all_docs = []
    file_count = 0
    
    for extension in FILE_EXTENSIONS:
        files = glob.glob(os.path.join(DATA_DIR, extension))
        for file_path in files:
            try:
                print(f"Loading {file_path}...")
                loader = TextLoader(file_path=file_path, encoding='utf-8')
                docs = loader.load()
                all_docs.extend(docs)
                file_count += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {file_count} files, {len(all_docs)} documents.")
    return all_docs

def get_or_create_vectordb():
    """Get existing vector DB or create new one"""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    # Check if persisted DB exists
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Loading existing vector database from {PERSIST_DIRECTORY}...")
        vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print("Vector database loaded.")
    else:
        # Create new database
        print("Creating new vector database...")
        data = load_documents()
        
        if not data:
            print("No data loaded. Please add .txt or .md files to the directory.")
            return None
        
        # Split and chunk
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        print(f"Created {len(chunks)} chunks.")
        
        # Create vector database with persistence
        print(f"Creating vector database with model {EMBEDDING_MODEL_NAME}...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        print("Vector database created and persisted.")
    
    return vector_db

def main():
    # 1. Get or create vector database
    vector_db = get_or_create_vectordb()
    if vector_db is None:
        return

    # 2. Retrieval Setup
    print(f"Initializing LLM {LLM_MODEL_NAME}...")
    llm = ChatOllama(model=LLM_MODEL_NAME)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT,
        parser_key="lines"
    )

    # 3. RAG Chain
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    
    IMPORTANT: Answer in the same language as the question."""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Interactive Query Loop
    print("\n" + "="*60)
    print("RAG System Ready! Type 'q' to quit.")
    print("="*60 + "\n")
    
    while True:
        query_text = input("\nYour question: ").strip()
        
        if query_text.lower() == 'q':
            print("Goodbye!")
            break
        
        if not query_text:
            print("Please enter a question.")
            continue
        
        print(f"\nProcessing...")
        
        try:
            result = chain.invoke(query_text)
            print("\n--- Answer ---")
            print(result)
            print("--------------")
        except Exception as e:
            print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()
