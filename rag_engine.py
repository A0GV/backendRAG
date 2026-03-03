
import os
import warnings
import glob

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


class RAGEngine:
    def __init__(self, data_dir="dataSet", embedding_model="text-embedding-3-small",
                 llm_model="openai/gpt-4o", persist_dir="./chroma_db", recursive=True):
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError(
                "OPENROUTER_API_KEY no encontrada. "
                "Por favor configura el archivo .env con tu API key de OpenRouter. "
                "Ver .env.example para más detalles."
            )

        self.data_dir = data_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.persist_dir = persist_dir
        self.recursive = recursive
        self.collection_name = "local-rag"

        self.vector_db = self._setup_vector_db()
        self.chain = self._setup_chain()

    def _load_documents(self):
        all_docs = []
        file_count = 0
        file_extensions = ["*.txt", "*.md"]

        for extension in file_extensions:
            if self.recursive:
                pattern = os.path.join(self.data_dir, "**", extension)
                files = glob.glob(pattern, recursive=True)
            else:
                files = glob.glob(os.path.join(self.data_dir, extension))

            for file_path in files:
                try:
                    loader = TextLoader(file_path=file_path, encoding='utf-8')
                    docs = loader.load()
                    all_docs.extend(docs)
                    file_count += 1
                except Exception as e:
                    print(f"Error cargando {file_path}: {e}")

        print(f"Cargados {file_count} archivos, {len(all_docs)} documentos.")
        return all_docs

    def _setup_vector_db(self):
        embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )

        if os.path.exists(self.persist_dir):
            print(f"Cargando base de datos vectorial existente desde {self.persist_dir}...")
            vector_db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
                collection_name=self.collection_name
            )
            print("Base de datos vectorial cargada. ✓")
        else:
            print("Creando nueva base de datos vectorial...")
            data = self._load_documents()

            if not data:
                raise ValueError("No se encontraron documentos. Agrega archivos .txt o .md al directorio.")

            print("Dividiendo texto en chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            print(f"Creados {len(chunks)} chunks.")

            print(f"Creando embeddings con modelo OpenAI {self.embedding_model}...")
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir
            )
            print("Base de datos vectorial creada y guardada. ✓")

        return vector_db

    def add_document(self, file_path):
        try:
            print(f"Procesando archivo para adición incremental: {file_path}")
            loader = TextLoader(file_path=file_path, encoding='utf-8')
            docs = loader.load()

            if not docs:
                print("El archivo está vacío o no se pudo cargar.")
                return 0

            print("Dividiendo texto en chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            print(f"Creados {len(chunks)} chunks del nuevo archivo.")

            if not chunks:
                print("No se generaron chunks (archivo muy pequeño?).")
                return 0

            print(f"Agregando chunks a ChromaDB en {self.persist_dir}...")
            self.vector_db.add_documents(chunks)
            print(f"✓ {len(chunks)} chunks agregados exitosamente.")
            return len(chunks)

        except Exception as e:
            print(f"Error agregando documento {file_path}: {e}")
            raise e

    def delete_document(self, file_path):
        try:
            print(f"Eliminando documento de la BD: {file_path}")

            results = self.vector_db.get(where={"source": file_path})
            if not results or not results['ids']:
                print(f"No se encontraron vectores para {file_path}")
                return False

            count = len(results['ids'])
            print(f"Encontrados {count} chunks para eliminar.")

            try:
                self.vector_db.delete(where={"source": file_path})
            except:
                if hasattr(self.vector_db, '_collection'):
                    self.vector_db._collection.delete(where={"source": file_path})
                else:
                    print("Error: No se encontró método delete compatible.")
                    return False

            print(f"✓ Documento eliminado de la BD exitosamente.")
            return True

        except Exception as e:
            print(f"Error eliminando documento {file_path}: {e}")
            raise e

    def _setup_chain(self):
        print(f"Inicializando LLM {self.llm_model} vía OpenRouter...")
        llm = ChatOpenAI(
            model=self.llm_model,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/backendRAG",
                "X-Title": "RAG System"
            }
        )

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""Eres un asistente de IA. Tu tarea es generar cinco
            versiones diferentes de la pregunta del usuario para recuperar documentos relevantes
            de una base de datos vectorial. Al generar múltiples perspectivas de la pregunta,
            ayudas a superar las limitaciones de la búsqueda por similitud basada en distancia.
            Proporciona estas preguntas alternativas separadas por saltos de línea.
            Pregunta original: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            self.vector_db.as_retriever(),
            llm,
            prompt=QUERY_PROMPT,
            parser_key="lines"
        )

        template = """Responde la pregunta basándote ÚNICAMENTE en el siguiente contexto:
        {context}
        Pregunta: {question}
        
        IMPORTANTE: Responde en el mismo idioma de la pregunta."""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain

    def ask(self, question):
        try:
            return self.chain.invoke(question)
        except Exception as e:
            return f"Error durante la ejecución: {e}"

    def update_database(self):
        import shutil
        import gc
        import time

        print(f"Actualizando base de datos vectorial en {self.persist_dir}...")

        if hasattr(self, 'vector_db'):
            self.vector_db = None
        if hasattr(self, 'chain'):
            self.chain = None

        gc.collect()
        time.sleep(1.0)

        if os.path.exists(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
                print(f"Directorio {self.persist_dir} eliminado.")
                time.sleep(1.0)
            except Exception as e:
                print(f"Error eliminando directorio: {e}")

        self.vector_db = self._setup_vector_db()
        self.chain = self._setup_chain()
        print("Base de datos actualizada exitosamente! ✓")
