import os
import sys
import warnings

# Suppress specific warnings if desired
warnings.filterwarnings("ignore")

# Fix for protobuf issue
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Configuration
DATA_PATH = "cat-facts.txt"
EMBEDDING_MODEL_NAME = "qwen3-embedding:8b"
LLM_MODEL_NAME = "qwen3-vl:8b"
COLLECTION_NAME = "local-rag"

def main():
    # 1. Check for file
    if not os.path.exists(DATA_PATH):
        print(f"Error: File '{DATA_PATH}' not found in the current directory.")
        print("Please ensure the file is present or update DATA_PATH in the script.")
        # Proceeding might fail, but let's try or return? 
        # Original script printed "Upload a PDF file" and then crashed on loader.
        return

    # 2. Load Data
    print(f"Loading {DATA_PATH}...")
    loader = TextLoader(file_path=DATA_PATH)
    data = loader.load()
    if not data:
        print("No data loaded.")
        return
    print(f"Loaded {len(data)} documents.") # TextLoader usually loads one document per file

    # 3. Split and chunk
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    print(f"Created {len(chunks)} chunks.")

    # 4. Vector Database
    print(f"Creating vector database with model {EMBEDDING_MODEL_NAME}...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL_NAME),
        collection_name=COLLECTION_NAME
    )
    print("Vector database created.")

    # 5. Retrieval Setup
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
        prompt=QUERY_PROMPT
    )

    # 6. RAG Chain
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Execute Query
    query_text = "tell me a random fact abou cats?"
    print(f"\nProcessing query: '{query_text}'")
    
    try:
        result = chain.invoke(query_text)
        print("\n--- Answer ---")
        print(result)
        print("--------------\n")
    except Exception as e:
        print(f"Error during execution: {e}")

    # Cleanup
    # Note: Deleting the collection immediately destroys the database. 
    # Uncomment the next line if you want to reset the DB every time.
    # vector_db.delete_collection()

if __name__ == "__main__":
    main()
