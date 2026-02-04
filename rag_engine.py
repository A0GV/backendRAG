import os
import warnings
import glob

warnings.filterwarnings("ignore")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


class RAGEngine:
    """Encapsulates RAG logic for reusable access across applications"""
    
    def __init__(self, data_dir="dataSet", embedding_model="qwen3-embedding:8b", 
                 llm_model="gemma3:4b", persist_dir="./chroma_db"):
        """Initialize RAG Engine
        
        Args:
            data_dir: Directory containing .txt and .md files
            embedding_model: Name of Ollama embedding model
            llm_model: Name of Ollama LLM model
            persist_dir: Directory to persist vector database
        """
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.persist_dir = persist_dir
        self.collection_name = "local-rag"
        
        # Initialize components
        self.vector_db = self._setup_vector_db()
        self.chain = self._setup_chain()
    
    def _load_documents(self):
        """Load all .txt and .md files from the data directory"""
        all_docs = []
        file_count = 0
        file_extensions = ["*.txt", "*.md"]
        
        for extension in file_extensions:
            files = glob.glob(os.path.join(self.data_dir, extension))
            for file_path in files:
                try:
                    loader = TextLoader(file_path=file_path, encoding='utf-8')
                    docs = loader.load()
                    all_docs.extend(docs)
                    file_count += 1
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {file_count} files, {len(all_docs)} documents.")
        return all_docs
    
    def _setup_vector_db(self):
        """Get existing vector DB or create new one"""
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        
        # Check if persisted DB exists
        if os.path.exists(self.persist_dir):
            print(f"Loading existing vector database from {self.persist_dir}...")
            vector_db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
                collection_name=self.collection_name
            )
            print("Vector database loaded.")
        else:
            # Create new database
            print("Creating new vector database...")
            data = self._load_documents()
            
            if not data:
                raise ValueError("No documents found. Please add .txt or .md files to the data directory.")
            
            # Split and chunk
            print("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            print(f"Created {len(chunks)} chunks.")
            
            # Create vector database with persistence
            print(f"Creating vector database with model {self.embedding_model}...")
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir
            )
            print("Vector database created and persisted.")
        
        return vector_db
    
    def _setup_chain(self):
        """Setup the RAG chain"""
        print(f"Initializing LLM {self.llm_model}...")
        llm = ChatOllama(model=self.llm_model)
        
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
            self.vector_db.as_retriever(),
            llm,
            prompt=QUERY_PROMPT,
            parser_key="lines"
        )
        
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
        
        return chain
    
    def ask(self, question):
        """Ask a question and get an answer
        
        Args:
            question: The question to ask
            
        Returns:
            The answer from the RAG chain
        """
        try:
            result = self.chain.invoke(question)
            return result
        except Exception as e:
            return f"Error during execution: {e}"
    
    def update_database(self):
        """Update the vector database with new documents"""
        print("Updating vector database...")
        self.vector_db.delete_collection()
        self.vector_db = self._setup_vector_db()
        self.chain = self._setup_chain()
        print("Database updated successfully!")
