"""
Motor RAG (Retrieval-Augmented Generation)

Este módulo encapsula toda la lógica de un sistema RAG:
1. Carga documentos (.txt, .md) desde un directorio
2. Los divide en chunks pequeños (1000 caracteres)
3. Crea embeddings (vectores numéricos) usando OpenAI
4. Almacena los vectores en ChromaDB
5. Permite hacer preguntas y recupera documentos relevantes
6. Genera respuestas usando un LLM basándose en el contexto recuperado
"""

import os
import warnings
import glob

# Cargar variables de entorno desde .env
from dotenv import load_dotenv
load_dotenv()

# Suprimir advertencias molestas
warnings.filterwarnings("ignore")

# Fix para compatibilidad de protobuf (evita errores con versiones antiguas/nuevas)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Importaciones de LangChain
from langchain_community.document_loaders import TextLoader  # Carga archivos de texto
from langchain_community.vectorstores import Chroma  # Base de datos vectorial
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # LLM y Embeddings de OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Divide texto en chunks
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate  # Plantillas de prompts
from langchain_core.output_parsers import StrOutputParser  # Parsea salida del LLM
from langchain_core.runnables import RunnablePassthrough  # Pasa datos sin modificar
from langchain_classic.retrievers.multi_query import MultiQueryRetriever  # Mejora búsqueda


class RAGEngine:
    """
    Motor RAG que encapsula toda la lógica de Retrieval-Augmented Generation.
    
    Flujo completo:
    1. __init__(): Inicializa el motor
       ├── _setup_vector_db(): Carga o crea la base de datos vectorial
       │   └── _load_documents(): Carga archivos .txt y .md
       └── _setup_chain(): Configura el pipeline RAG (retriever → prompt → LLM)
    
    2. ask(question): Procesa una pregunta
       └── chain.invoke(): Ejecuta todo el flujo RAG
    """
    
    def __init__(self, data_dir="dataSet", embedding_model="text-embedding-3-small", 
                 llm_model="openai/gpt-4o", persist_dir="./chroma_db", recursive=True):
        """
        Inicializa el motor RAG.
        
        Args:
            data_dir: Directorio que contiene archivos .txt y .md
            embedding_model: Nombre del modelo de embeddings en OpenAI (default: text-embedding-3-small)
            llm_model: Nombre del modelo LLM en OpenRouter (default: openai/gpt-4o)
            persist_dir: Directorio donde se guarda la base de datos vectorial
            recursive: Si es True, busca archivos en subdirectorios. Si es False, solo en data_dir.
        
        Flujo de inicialización:
        1. Valida que existe la API key de OpenRouter
        2. Guarda la configuración
        3. Llama a _setup_vector_db() para cargar/crear la BD vectorial
        4. Llama a _setup_chain() para configurar el pipeline RAG
        """
        # Validar que existe la API key
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError(
                "OPENROUTER_API_KEY no encontrada. "
                "Por favor configura el archivo .env con tu API key de OpenRouter. "
                "Ver .env.example para más detalles."
            )
        
        # Guardar configuración
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.persist_dir = persist_dir
        self.recursive = recursive
        self.collection_name = "local-rag"
        
        # PASO 1: Configurar la base de datos vectorial
        # Si existe en disco, la carga; si no, la crea desde cero
        self.vector_db = self._setup_vector_db()
        
        # PASO 2: Configurar el chain RAG (retriever → prompt → LLM → parser)
        self.chain = self._setup_chain()
    
    def _load_documents(self):
        """
        Carga todos los archivos .txt y .md del directorio de datos.
        
        Flujo:
        1. Busca todos los archivos con extensión .txt y .md (recursivamente si recursive=True)
        2. Los carga uno por uno usando TextLoader
        3. Retorna una lista de objetos Document
        
        Returns:
            Lista de documentos cargados
        """
        all_docs = []
        file_count = 0
        file_extensions = ["*.txt", "*.md"]  # Extensiones soportadas
        
        # Iterar por cada extensión de archivo
        for extension in file_extensions:
            # Buscar todos los archivos con esta extensión
            if self.recursive:
                # Búsqueda recursiva usando glob y **
                pattern = os.path.join(self.data_dir, "**", extension)
                files = glob.glob(pattern, recursive=True)
            else:
                # Búsqueda solo en el nivel superior
                files = glob.glob(os.path.join(self.data_dir, extension))
            
            # Cargar cada archivo encontrado
            for file_path in files:
                try:
                    # TextLoader lee el archivo como texto plano
                    loader = TextLoader(file_path=file_path, encoding='utf-8')
                    docs = loader.load()  # Retorna lista de Document objects
                    all_docs.extend(docs)  # Agregar a la lista principal
                    file_count += 1
                except Exception as e:
                    print(f"Error cargando {file_path}: {e}")
        
        print(f"Cargados {file_count} archivos, {len(all_docs)} documentos.")
        return all_docs
    
    def _setup_vector_db(self):
        """
        Carga la base de datos vectorial existente o crea una nueva.
        
        Flujo:
        A. Si existe chroma_db/:
           1. Cargar la BD desde disco (muy rápido)
           2. Usar los embeddings ya calculados
        
        B. Si NO existe:
           1. Cargar documentos con _load_documents()
           2. Dividir en chunks de 1000 caracteres con overlap de 100
           3. Crear embeddings para cada chunk (texto → vector usando OpenAI)
           4. Guardar en ChromaDB en disco para uso futuro
        
        Returns:
            Objeto Chroma (base de datos vectorial)
        """
        # Crear función de embeddings (convierte texto en vectores numéricos usando OpenAI)
        embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        # OPCIÓN A: Si ya existe la BD en disco
        if os.path.exists(self.persist_dir):
            print(f"Cargando base de datos vectorial existente desde {self.persist_dir}...")
            vector_db = Chroma(
                persist_directory=self.persist_dir,  # Cargar desde disco
                embedding_function=embeddings,  # Función para crear embeddings
                collection_name=self.collection_name  # Nombre de la colección
            )
            print("Base de datos vectorial cargada. ✓")
        
        # OPCIÓN B: Si NO existe, crearla desde cero
        else:
            print("Creando nueva base de datos vectorial...")
            
            # PASO 1: Cargar documentos del directorio
            data = self._load_documents()
            
            if not data:
                raise ValueError("No se encontraron documentos. Agrega archivos .txt o .md al directorio.")
            
            # PASO 2: Dividir texto en chunks pequeños
            # - chunk_size=1000: Cada chunk tiene máximo 1000 caracteres
            # - chunk_overlap=100: Últimos 100 caracteres se repiten en siguiente chunk
            #   (mantiene contexto entre chunks)
            print("Dividiendo texto en chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(data)
            print(f"Creados {len(chunks)} chunks.")
            
            # PASO 3: Crear base de datos vectorial
            # - Convierte cada chunk en un vector numérico (embedding) usando OpenAI
            # - Guarda los vectores + texto original en ChromaDB
            # - Persiste en disco para reutilizar
            print(f"Creando embeddings con modelo OpenAI {self.embedding_model}...")
            vector_db = Chroma.from_documents(
                documents=chunks,  # Los chunks a procesar
                embedding=embeddings,  # Función para crear embeddings
                collection_name=self.collection_name,  # Nombre de la colección
                persist_directory=self.persist_dir  # Guardar en disco
            )
            print("Base de datos vectorial creada y guardada. ✓")
        
        return vector_db
    
    def add_document(self, file_path):
        """
        Agrega un documento individual a la base de datos existente.
        
        Este método NO borra la BD existente, solo agrega el nuevo documento.
        
        Args:
            file_path: Ruta completa al archivo .txt o .md
        
        Returns:
            int: Número de chunks agregados
        """
        try:
            print(f"Procesando archivo para adición incremental: {file_path}")
            
            # 1. Cargar el documento
            loader = TextLoader(file_path=file_path, encoding='utf-8')
            docs = loader.load()
            
            if not docs:
                print("El archivo está vacío o no se pudo cargar.")
                return 0
            
            # 2. Dividir en chunks
            print("Dividiendo texto en chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(docs)
            print(f"Creados {len(chunks)} chunks del nuevo archivo.")
            
            if not chunks:
                print("No se generaron chunks (archivo muy pequeño?).")
                return 0
            
            # 3. Agregar a la base de datos existente (sin borrar)
            # ChromaDB maneja automáticamente la generación de IDs si no se proveen
            print(f"Agregando chunks a ChromaDB en {self.persist_dir}...")
            self.vector_db.add_documents(chunks)
            print(f"✓ {len(chunks)} chunks agregados exitosamente.")
            
            return len(chunks)
            
        except Exception as e:
            print(f"Error agregando documento {file_path}: {e}")
            raise e

    def delete_document(self, file_path):
        """
        Elimina un documento de la base de datos vectorial.
        
        Busca todos los chunks que tengan metadata 'source' == file_path
        y los elimina.
        
        Args:
            file_path: Ruta completa al archivo .txt o .md
            
        Returns:
            bool: True si se eliminó algo, False si no
        """
        try:
            print(f"Eliminando documento de la BD: {file_path}")
            
            # ChromaDB permite borrar por metadata usando 'where'
            # source es el metadata que LangChain pone por defecto con el path
            # IMPORTANTE: El path debe ser EXACTAMENTE el mismo que se usó al cargar
            
            # Primero verificamos si hay algo que borrar (opcional, pero bueno para logs)
            results = self.vector_db.get(where={"source": file_path})
            if not results or not results['ids']:
                print(f"No se encontraron vectores para {file_path}")
                return False
            
            count = len(results['ids'])
            print(f"Encontrados {count} chunks para eliminar.")
            
            # Ejecutar borrado
            # IMPORTANTE: Usamos delete() que es método público de Chroma (vectorstore)
            # En LangChain 0.2+, Chroma.delete() acepta 'where'
            # Si falla, intentamos _collection.delete()
            try:
                self.vector_db.delete(where={"source": file_path})
            except:
                # Fallback para versiones antiguas o diferente implementación
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
        """
        Configura el pipeline RAG (chain).
        
        Flujo del chain cuando se hace una pregunta:
        1. ENTRADA: Pregunta del usuario ("¿Dato de gatos?")
        
        2. MultiQueryRetriever:
           ├── Genera 5 versiones de la pregunta usando el LLM
           ├── Busca documentos similares en ChromaDB para cada versión
           └── Retorna los documentos más relevantes (deduplicados)
        
        3. Prompt:
           ├── Combina el contexto recuperado + pregunta original
           └── Crea un mensaje estructurado para el LLM
        
        4. LLM (ChatOpenAI vía OpenRouter):
           ├── Procesa el prompt
           └── Genera respuesta basada en el contexto
        
        5. StrOutputParser:
           └── Extrae solo el texto de la respuesta
        
        6. SALIDA: Respuesta final al usuario
        
        Returns:
            Chain completo (pipeline) listo para usar
        """
        # Inicializar el modelo de lenguaje (OpenAI vía OpenRouter)
        print(f"Inicializando LLM {self.llm_model} vía OpenRouter...")
        llm = ChatOpenAI(
            model=self.llm_model,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/backendRAG",  # Opcional: para analytics
                "X-Title": "RAG System"  # Opcional: nombre de tu app
            }
        )
        
        # COMPONENTE 1: MultiQueryRetriever
        # Genera múltiples versiones de la pregunta para mejorar la búsqueda
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""Eres un asistente de IA. Tu tarea es generar cinco
            versiones diferentes de la pregunta del usuario para recuperar documentos relevantes
            de una base de datos vectorial. Al generar múltiples perspectivas de la pregunta,
            ayudas a superar las limitaciones de la búsqueda por similitud basada en distancia.
            Proporciona estas preguntas alternativas separadas por saltos de línea.
            Pregunta original: {question}""",
        )
        
        # Crear el retriever que:
        # 1. Toma la pregunta original
        # 2. Genera 5 versiones usando el LLM
        # 3. Busca documentos para cada versión
        # 4. Retorna los más relevantes
        retriever = MultiQueryRetriever.from_llm(
            self.vector_db.as_retriever(),  # Retriever base de ChromaDB
            llm,  # LLM para generar las 5 versiones
            prompt=QUERY_PROMPT,  # Template para generar versiones
            parser_key="lines"  # Parsear salida línea por línea
        )
        
        # COMPONENTE 2: Prompt Template para el LLM final
        # Este prompt combina el contexto recuperado con la pregunta
        template = """Responde la pregunta basándote ÚNICAMENTE en el siguiente contexto:
        {context}
        Pregunta: {question}
        
        IMPORTANTE: Responde en el mismo idioma de la pregunta."""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # COMPONENTE 3: Ensamblar el Chain (Pipeline)
        # Flujo: Pregunta → Retriever → Prompt → LLM → Parser → Respuesta
        chain = (
            # Paso 1: Preparar entrada
            {
                "context": retriever,  # Busca contexto relevante
                "question": RunnablePassthrough()  # Pasa la pregunta sin cambios
            }
            # Paso 2: Crear prompt con contexto + pregunta
            | prompt
            # Paso 3: Enviar al LLM para generar respuesta
            | llm
            # Paso 4: Extraer solo el texto de la respuesta
            | StrOutputParser()
        )
        
        return chain
    
    def ask(self, question):
        """
        Hace una pregunta al sistema RAG y obtiene una respuesta.
        
        Flujo completo cuando llamas a ask("¿Dato de gatos?"):
        1. chain.invoke() ejecuta todo el pipeline:
           ├── Retriever genera 5 versiones de la pregunta
           ├── Busca documentos relevantes en ChromaDB
           ├── Construye prompt con contexto + pregunta
           ├── LLM genera respuesta basada en contexto
           └── Parser extrae el texto de la respuesta
        
        2. Retorna la respuesta final
        
        Args:
            question: La pregunta a hacer (string)
            
        Returns:
            La respuesta generada por el LLM (string)
        """
        try:
            # Ejecutar todo el chain RAG
            result = self.chain.invoke(question)
            return result
        except Exception as e:
            return f"Error durante la ejecución: {e}"
    
    def update_database(self):
        """
        Actualiza la base de datos vectorial con nuevos documentos.
        
        ATENCIÓN: Este método BORRA toda la colección existente y la recrea desde cero.
        Esto significa:
        - Se pierden todos los embeddings anteriores
        - Se vuelven a procesar TODOS los archivos (no solo los nuevos)
        - Tarda más tiempo en ejecutar
        
        Flujo:
        1. Borra completamente el directorio chroma_db/ del disco
        2. Vuelve a ejecutar _setup_vector_db() (carga archivos, crea chunks, embeddings)
        3. Reconfigura el chain con la nueva BD
        
        Uso:
            rag = RAGEngine()
            # ... agregas nuevos archivos a dataSet/ ...
            rag.update_database()  # Recrea la BD con todos los archivos
        """
        import shutil
        import gc
        
        print(f"Actualizando base de datos vectorial en {self.persist_dir}...")
        
        # PASO 1: Liberar recursos y borrar directorio
        # Intentar liberar el objeto vector_db para soltar locks
        if hasattr(self, 'vector_db'):
            self.vector_db = None
        
        # IMPORTANTE: También liberar el chain, ya que tiene referencias al vector_db
        if hasattr(self, 'chain'):
            self.chain = None
            
        gc.collect()  # Forzar recolección de basura
        
        # Esperar un momento para que el OS libere los archivos
        import time
        time.sleep(1.0)
        
        if os.path.exists(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
                print(f"Directorio {self.persist_dir} eliminado.")
                # Esperar otro momento para asegurar que el directorio se eliminó
                time.sleep(1.0)
            except Exception as e:
                print(f"Error eliminando directorio: {e}")
                # Si falla borrar, intentamos continuar (puede que falle al crear)
        
        # PASO 2: Recrear la BD desde cero (lee todos los archivos del directorio)
        self.vector_db = self._setup_vector_db()
        
        # PASO 3: Reconfigurar el chain con la nueva BD
        self.chain = self._setup_chain()
        
        print("Base de datos actualizada exitosamente! ✓")
