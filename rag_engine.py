"""
Motor RAG (Retrieval-Augmented Generation)

Este módulo encapsula toda la lógica de un sistema RAG:
1. Carga documentos (.txt, .md) desde un directorio
2. Los divide en chunks pequeños (1000 caracteres)
3. Crea embeddings (vectores numéricos) usando Ollama
4. Almacena los vectores en ChromaDB
5. Permite hacer preguntas y recupera documentos relevantes
6. Genera respuestas usando un LLM basándose en el contexto recuperado
"""

import os
import warnings
import glob

# Suprimir advertencias molestas
warnings.filterwarnings("ignore")

# Fix para compatibilidad de protobuf (evita errores con versiones antiguas/nuevas)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Importaciones de LangChain
from langchain_community.document_loaders import TextLoader  # Carga archivos de texto
from langchain_community.vectorstores import Chroma  # Base de datos vectorial
from langchain_ollama import OllamaEmbeddings  # Genera embeddings con Ollama
from langchain_ollama.chat_models import ChatOllama  # Modelo de lenguaje
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
    
    def __init__(self, data_dir="dataSet", embedding_model="qwen3-embedding:8b", 
                 llm_model="llama3.1:latest", persist_dir="./chroma_db"):
        """
        Inicializa el motor RAG.
        
        Args:
            data_dir: Directorio que contiene archivos .txt y .md
            embedding_model: Nombre del modelo de embeddings en Ollama
            llm_model: Nombre del modelo LLM en Ollama
            persist_dir: Directorio donde se guarda la base de datos vectorial
        
        Flujo de inicialización:
        1. Guarda la configuración
        2. Llama a _setup_vector_db() para cargar/crear la BD vectorial
        3. Llama a _setup_chain() para configurar el pipeline RAG
        """
        # Guardar configuración
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.persist_dir = persist_dir
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
        1. Busca todos los archivos con extensión .txt y .md
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
           3. Crear embeddings para cada chunk (texto → vector de 1024 números)
           4. Guardar en ChromaDB en disco para uso futuro
        
        Returns:
            Objeto Chroma (base de datos vectorial)
        """
        # Crear función de embeddings (convierte texto en vectores numéricos)
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        
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
            # - Convierte cada chunk en un vector numérico (embedding)
            # - Guarda los vectores + texto original en ChromaDB
            # - Persiste en disco para reutilizar
            print(f"Creando base de datos vectorial con modelo {self.embedding_model}...")
            vector_db = Chroma.from_documents(
                documents=chunks,  # Los chunks a procesar
                embedding=embeddings,  # Función para crear embeddings
                collection_name=self.collection_name,  # Nombre de la colección
                persist_directory=self.persist_dir  # Guardar en disco
            )
            print("Base de datos vectorial creada y guardada. ✓")
        
        return vector_db
    
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
        
        4. LLM (ChatOllama):
           ├── Procesa el prompt
           └── Genera respuesta basada en el contexto
        
        5. StrOutputParser:
           └── Extrae solo el texto de la respuesta
        
        6. SALIDA: Respuesta final al usuario
        
        Returns:
            Chain completo (pipeline) listo para usar
        """
        # Inicializar el modelo de lenguaje
        print(f"Inicializando LLM {self.llm_model}...")
        llm = ChatOllama(model=self.llm_model)
        
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
        1. Borra la colección actual de ChromaDB
        2. Vuelve a ejecutar _setup_vector_db() (carga archivos, crea chunks, embeddings)
        3. Reconfigura el chain con la nueva BD
        
        Uso:
            rag = RAGEngine()
            # ... agregas nuevos archivos a dataSet/ ...
            rag.update_database()  # Recrea la BD con todos los archivos
        """
        print("Actualizando base de datos vectorial...")
        
        # PASO 1: Borrar la colección existente
        self.vector_db.delete_collection()
        
        # PASO 2: Recrear la BD desde cero (lee todos los archivos del directorio)
        self.vector_db = self._setup_vector_db()
        
        # PASO 3: Reconfigurar el chain con la nueva BD
        self.chain = self._setup_chain()
        
        print("Base de datos actualizada exitosamente! ✓")
