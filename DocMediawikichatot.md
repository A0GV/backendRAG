# Documentación del Sistema RAG Multi-Contexto

## Índice
1. [Descripción General](#descripción-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Componentes Principales](#componentes-principales)
4. [Flujo de Trabajo Detallado](#flujo-de-trabajo-detallado)
5. [API REST (Server)](#api-rest-server)
6. [Configuración y Uso](#configuración-y-uso)
7. [Migración a Otras APIs LLM](#migración-a-otras-apis-llm)

---

## Descripción General

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** multi-contexto que permite consultar información de documentos con diferentes niveles de acceso. Utiliza:

- **LangChain**: Framework para orquestar componentes LLM
- **Ollama**: Servicio local de modelos LLM y embeddings
- **ChromaDB**: Base de datos vectorial para almacenar embeddings
- **Flask**: API REST para exponer servicios

### Características Principales
- ✅ Gestión de múltiples contextos (cliente vs distribuidor)
- ✅ Búsqueda semántica avanzada con MultiQueryRetriever
- ✅ Persistencia de embeddings en disco
- ✅ API REST para integración con frontends
- ✅ Actualización dinámica de bases de datos

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    USUARIO / FRONTEND                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│               Flask API REST (server/)                       │
│  - routes.py: Endpoints /api/ask, /api/update               │
│  - run.py: Servidor Flask con CORS                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│         MultiRAGEngine (multi_rag_engine.py)                │
│  Orquestador de múltiples contextos:                        │
│  - rag_cliente: Acceso restringido (dataSet/datasetGen/)    │
│  - rag_distribuidor: Acceso completo (dataSet/)             │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
┌──────────────────┐              ┌──────────────────┐
│  RAGEngine       │              │  RAGEngine       │
│  (Cliente)       │              │  (Distribuidor)  │
│                  │              │                  │
│ chroma_cliente/  │              │ chroma_dist../   │
└──────────────────┘              └──────────────────┘
        ↓                                   ↓
┌─────────────────────────────────────────────────────────────┐
│              ChromaDB (Base de Datos Vectorial)             │
│  - Almacena embeddings (vectores numéricos)                 │
│  - Búsqueda por similitud coseno                            │
└─────────────────────────────────────────────────────────────┘
        ↓                                   ↓
┌─────────────────────────────────────────────────────────────┐
│                    Ollama (Servicio Local)                  │
│  - Modelo de Embeddings: qwen3-embedding:8b                 │
│  - Modelo LLM: llama3.1:latest                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Componentes Principales

### 1. **rag_engine.py** - Motor RAG Base

**Responsabilidad**: Encapsula toda la lógica RAG para un contexto específico.

#### Clase `RAGEngine`

**Atributos principales**:
```python
data_dir         # Directorio con archivos .txt y .md
embedding_model  # Modelo de embeddings (default: qwen3-embedding:8b)
llm_model        # Modelo LLM (default: llama3.1:latest)
persist_dir      # Directorio para persistir ChromaDB
recursive        # Si busca archivos recursivamente
vector_db        # Instancia de Chroma
chain            # Pipeline RAG completo
```

**Métodos principales**:

##### `__init__(data_dir, embedding_model, llm_model, persist_dir, recursive)`
Inicializa el motor RAG:
1. Carga o crea la base de datos vectorial (`_setup_vector_db()`)
2. Configura el pipeline RAG (`_setup_chain()`)

##### `_load_documents()`
Carga archivos .txt y .md del directorio:
- Soporta búsqueda recursiva con `glob(**/*.txt)`
- Maneja errores de codificación UTF-8
- Retorna lista de objetos `Document` de LangChain

##### `_setup_vector_db()`
Gestiona la base de datos vectorial:

**Si existe `persist_dir`**:
1. Carga embeddings desde disco (rápido)

**Si NO existe**:
1. Carga documentos con `_load_documents()`
2. Divide en chunks de 1000 caracteres (overlap 100)
3. Genera embeddings para cada chunk
4. Persiste en disco

**Chunk Strategy**:
```
Documento Original (5000 caracteres)
↓
RecursiveCharacterTextSplitter
├── chunk_size = 1000
└── chunk_overlap = 100
↓
[Chunk 1: 0-1000]
[Chunk 2: 900-1900]  ← 100 caracteres se solapan
[Chunk 3: 1800-2800]
...
```

##### `_setup_chain()`
Configura el pipeline RAG completo:

```python
Pipeline:
Pregunta → MultiQueryRetriever → Prompt → LLM → StrOutputParser → Respuesta

1. MultiQueryRetriever:
   - Genera 5 versiones de la pregunta
   - Busca documentos para cada versión
   - Deduplica resultados

2. Prompt Template:
   - Combina contexto recuperado + pregunta
   - Instrucciones en español/mismo idioma

3. ChatOllama:
   - Procesa prompt
   - Genera respuesta basada en contexto

4. StrOutputParser:
   - Extrae texto de la respuesta
```

**Ejemplo de MultiQuery**:
```
Pregunta Original: "¿Cómo configurar autenticación?"

MultiQueryRetriever genera:
1. "¿Cuál es el proceso de configuración de autenticación?"
2. "¿Qué pasos seguir para autenticar usuarios?"
3. "¿Cómo implementar login y autenticación?"
4. "¿Dónde configurar credenciales de autenticación?"
5. "¿Qué métodos de autenticación están disponibles?"

→ Busca en ChromaDB con las 5 versiones
→ Combina y deduplica resultados
→ Retorna los 4-5 documentos más relevantes
```

##### `ask(question)`
Ejecuta el pipeline RAG completo:
```python
result = chain.invoke(question)
```

##### `update_database()`
Actualiza la base de datos:
1. Libera recursos (vector_db, chain)
2. Borra directorio `persist_dir`
3. Recrea desde cero con `_setup_vector_db()`

⚠️ **IMPORTANTE**: Reprocesa TODOS los archivos, no solo nuevos.

---

### 2. **multi_rag_engine.py** - Orquestador Multi-Contexto

**Responsabilidad**: Gestiona múltiples instancias de RAGEngine para diferentes niveles de acceso.

#### Clase `MultiRAGEngine`

**Estrategia de Contextos**:

| Contexto      | Directorio              | Recursive | Persist Dir           |
|---------------|-------------------------|-----------|----------------------|
| **cliente**   | `dataSet/datasetGen/`   | False     | `./chroma_cliente`   |
| **distribuidor** | `dataSet/`           | True      | `./chroma_distribuidor` |

**Casos de Uso**:
- **Cliente**: Solo accede a información pública/restringida en `datasetGen/`
- **Distribuidor**: Accede a TODO (datasetGen/ + otros documentos en dataSet/)

**Métodos principales**:

##### `__init__()`
Inicializa ambas instancias de RAGEngine:
```python
self.rag_distribuidor = RAGEngine(data_dir="dataSet", recursive=True)
self.rag_cliente = RAGEngine(data_dir="dataSet/datasetGen", recursive=False)
```

##### `get_engine(db_type)`
Retorna la instancia correcta:
```python
if db_type == "distribuidor":
    return self.rag_distribuidor
else:
    return self.rag_cliente
```

##### `ask(question, db_type)`
Enruta la pregunta al motor correspondiente:
```python
engine = self.get_engine(db_type)
return engine.ask(question)
```

##### `update_database()`
Actualiza AMBAS bases de datos:
1. Libera recursos de ambas instancias
2. Borra ambos directorios de persistencia
3. Recrea las instancias desde cero

---

### 3. **tre.py** - Interfaz CLI Interactiva

**Responsabilidad**: Proporciona interfaz de línea de comandos para testing.

**Comandos disponibles**:
```
q                 → Salir
update            → Actualizar todas las bases de datos
use cliente       → Cambiar a contexto cliente
use distribuidor  → Cambiar a contexto distribuidor
<pregunta>        → Hacer una consulta
```

**Flujo de interacción**:
```python
1. Inicializa MultiRAGEngine
2. Loop infinito:
   - Lee comando del usuario
   - Si es comando especial → ejecuta acción
   - Si es pregunta → llama a rag.ask(query, db_type)
   - Muestra respuesta
```

---

### 4. **server/** - API REST con Flask

#### Estructura:
```
server/
├── run.py              # Punto de entrada del servidor
├── app/
    ├── __init__.py     # Factory de aplicación Flask
    └── routes.py       # Definición de endpoints
```

#### **run.py**
```python
# Configura sys.path para importar módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Crea app Flask
app = create_app()

# Habilita CORS para peticiones cross-origin
CORS(app)

# Inicia servidor en 0.0.0.0:5000
```

#### **app/__init__.py**
Factory pattern para crear aplicación Flask:
```python
def create_app():
    app = Flask(__name__)
    app.register_blueprint(api)  # Registra rutas
    return app
```

#### **app/routes.py**

**Singleton de MultiRAGEngine**:
```python
rag_engine = MultiRAGEngine()  # Se inicializa una vez al importar
```

**Endpoints**:

##### `GET /`
Health check:
```json
Response: {"status": "ok"} o {"status": "error"}
```

##### `POST /api/ask`
Consulta al sistema RAG:

**Request**:
```json
{
  "question": "¿Cómo configurar X?",
  "db_type": "cliente"  // Opcional: "cliente" (default) o "distribuidor"
}
```

**Response Success**:
```json
{
  "success": true,
  "answer": "Para configurar X debes...",
  "db_used": "cliente"
}
```

**Response Error**:
```json
{
  "success": false,
  "answer": "Error message"
}
```

##### `POST /api/update`
Actualiza todas las bases de datos:

**Request**: Vacío

**Response**:
```json
{
  "success": true,
  "message": "All databases updated successfully"
}
```

---

## Flujo de Trabajo Detallado

### A. Inicialización del Sistema

```
1. Usuario ejecuta: python server/run.py
   ↓
2. Flask importa routes.py
   ↓
3. routes.py ejecuta:
   rag_engine = MultiRAGEngine()
   ↓
4. MultiRAGEngine.__init__():
   ├── Crea rag_distribuidor:
   │   ├── Verifica si existe ./chroma_distribuidor/
   │   ├── SI EXISTE: Carga embeddings desde disco
   │   └── NO EXISTE:
   │       ├── Lee todos los .txt/.md de dataSet/ (recursivo)
   │       ├── Divide en chunks (1000 chars, overlap 100)
   │       ├── Genera embeddings con qwen3-embedding:8b
   │       └── Persiste en ./chroma_distribuidor/
   │
   └── Crea rag_cliente:
       └── Mismo proceso pero solo para dataSet/datasetGen/
   ↓
5. Servidor Flask listo en 0.0.0.0:5000
```

### B. Procesamiento de una Consulta

```
1. Cliente HTTP envía POST a /api/ask:
   {
     "question": "¿Qué es un webhook?",
     "db_type": "cliente"
   }
   ↓
2. routes.ask() recibe petición
   ↓
3. Valida datos:
   - question existe y no está vacío
   - db_type es "cliente" o "distribuidor"
   ↓
4. Llama: rag_engine.ask(question, db_type="cliente")
   ↓
5. MultiRAGEngine.ask():
   - Selecciona self.rag_cliente
   - Llama rag_cliente.ask(question)
   ↓
6. RAGEngine.ask():
   chain.invoke("¿Qué es un webhook?")
   ↓
7. Pipeline RAG:
   
   a) MultiQueryRetriever:
      ├── Genera 5 variaciones de la pregunta
      ├── Busca en ChromaDB con cada variación
      ├── Similarity Search (cosine distance)
      └── Retorna top 4-5 documentos
   
   b) Contexto recuperado:
      [
        "Un webhook es un mecanismo de notificación...",
        "Los webhooks permiten que aplicaciones...",
        "Para configurar un webhook necesitas..."
      ]
   
   c) Prompt Template:
      """
      Responde basándote ÚNICAMENTE en el siguiente contexto:
      [contexto recuperado]
      
      Pregunta: ¿Qué es un webhook?
      """
   
   d) ChatOllama (llama3.1):
      - Procesa el prompt
      - Genera respuesta coherente
   
   e) StrOutputParser:
      - Extrae texto de la respuesta
   ↓
8. Retorna respuesta al cliente:
   {
     "success": true,
     "answer": "Un webhook es un mecanismo de notificación HTTP...",
     "db_used": "cliente"
   }
```

### C. Actualización de Base de Datos

```
1. Cliente envía POST a /api/update
   ↓
2. routes.update_database():
   rag_engine.update_database()
   ↓
3. MultiRAGEngine.update_database():
   
   Para cada instancia (cliente y distribuidor):
   
   a) Libera recursos:
      - self.rag_X.vector_db = None
      - self.rag_X.chain = None
      - gc.collect()
   
   b) Espera 1 segundo
   
   c) Borra directorio:
      - shutil.rmtree(./chroma_X/)
   
   d) Espera 1 segundo
   
   e) Recrea instancia:
      - RAGEngine(...)
      - Lee TODOS los archivos
      - Genera embeddings
      - Persiste en disco
   ↓
4. Retorna confirmación al cliente
```

---

## API REST (Server)

### Características Técnicas

**Framework**: Flask 2.x  
**CORS**: Habilitado para todos los orígenes  
**Puerto**: 5000  
**Host**: 0.0.0.0 (accesible desde red local)  
**Debug Mode**: True (solo para desarrollo)

### Ejemplo de Uso con cURL

```bash
# Health check
curl http://localhost:5000/

# Consulta como cliente
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cómo funciona la autenticación?",
    "db_type": "cliente"
  }'

# Consulta como distribuidor
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Configuración avanzada del sistema?",
    "db_type": "distribuidor"
  }'

# Actualizar bases de datos
curl -X POST http://localhost:5000/api/update
```

### Ejemplo con JavaScript (Fetch API)

```javascript
// Consulta al sistema RAG
async function askRAG(question, dbType = 'cliente') {
  const response = await fetch('http://localhost:5000/api/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question: question,
      db_type: dbType
    })
  });
  
  const data = await response.json();
  
  if (data.success) {
    console.log(`Respuesta (${data.db_used}):`, data.answer);
  } else {
    console.error('Error:', data.answer);
  }
}

// Uso
askRAG('¿Qué es un webhook?', 'cliente');
```

---

## Configuración y Uso

### Requisitos Previos

1. **Ollama instalado y corriendo**:
```bash
# Verificar que Ollama está corriendo
curl http://localhost:11434/api/tags

# Descargar modelos necesarios
ollama pull llama3.1:latest
ollama pull qwen3-embedding:8b
```

2. **Python 3.8+**

3. **Dependencias instaladas**:
```bash
pip install -r requirements.txt
```

### Estructura de Directorios

```
backendRAG/
├── dataSet/                # Documentos para distribuidor
│   ├── doc1.txt
│   ├── doc2.md
│   └── datasetGen/         # Documentos para cliente
│       ├── public1.txt
│       └── public2.md
├── chroma_distribuidor/    # BD vectorial distribuidor (auto-generado)
├── chroma_cliente/         # BD vectorial cliente (auto-generado)
├── rag_engine.py
├── multi_rag_engine.py
├── tre.py
├── server/
│   ├── run.py
│   └── app/
│       ├── __init__.py
│       └── routes.py
└── requirements.txt
```

### Modo 1: CLI Interactivo

```bash
python tre.py
```

### Modo 2: API REST

```bash
# Iniciar servidor
cd server
python run.py

# En otro terminal, hacer peticiones
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tu pregunta aquí", "db_type": "cliente"}'
```

---

## Migración a Otras APIs LLM

### Conceptos Clave a Cambiar

El sistema actual usa **Ollama** (servicio local) para:
1. **Embeddings**: Convertir texto a vectores
2. **LLM**: Generar respuestas

Para migrar a **Gemini**, **OpenAI**, o cualquier otra API, debes cambiar:

### 1. Dependencias (requirements.txt)

**Agregar** según la API:

#### Para Google Gemini:
```txt
# Agregar
langchain-google-genai>=2.0.0
google-generativeai>=0.8.0

# Eliminar (ya no necesarios)
# langchain-ollama==1.0.1
```

#### Para OpenAI (ChatGPT):
```txt
# Agregar
langchain-openai>=0.2.0
openai>=1.0.0

# Eliminar (ya no necesarios)
# langchain-ollama==1.0.1
```

#### Para Azure OpenAI:
```txt
# Agregar
langchain-openai>=0.2.0
openai>=1.0.0
```

### 2. Cambios en `rag_engine.py`

#### Ubicación: Líneas 23-27 (Imports)

**Actual (Ollama)**:
```python
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
```

**Para Gemini**:
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
```

**Para OpenAI**:
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
```

**Para Azure OpenAI**:
```python
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
```

---

#### Ubicación: Línea 49-50 (Constructor)

**Actual (Ollama)**:
```python
def __init__(self, data_dir="dataSet", embedding_model="qwen3-embedding:8b", 
             llm_model="llama3.1:latest", persist_dir="./chroma_db", recursive=True):
```

**Para Gemini**:
```python
def __init__(self, data_dir="dataSet", 
             embedding_model="models/embedding-001",  # Modelo de embeddings de Gemini
             llm_model="gemini-1.5-pro",  # O gemini-1.5-flash
             api_key=None,  # Nuevo parámetro
             persist_dir="./chroma_db", recursive=True):
    
    # Guardar API key
    self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not self.api_key:
        raise ValueError("GOOGLE_API_KEY no configurada")
```

**Para OpenAI**:
```python
def __init__(self, data_dir="dataSet", 
             embedding_model="text-embedding-3-large",  # O text-embedding-ada-002
             llm_model="gpt-4-turbo",  # O gpt-3.5-turbo
             api_key=None,
             persist_dir="./chroma_db", recursive=True):
    
    self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not self.api_key:
        raise ValueError("OPENAI_API_KEY no configurada")
```

---

#### Ubicación: Línea 141 (_setup_vector_db)

**Actual (Ollama)**:
```python
embeddings = OllamaEmbeddings(model=self.embedding_model)
```

**Para Gemini**:
```python
embeddings = GoogleGenerativeAIEmbeddings(
    model=self.embedding_model,
    google_api_key=self.api_key
)
```

**Para OpenAI**:
```python
embeddings = OpenAIEmbeddings(
    model=self.embedding_model,
    openai_api_key=self.api_key
)
```

**Para Azure OpenAI**:
```python
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="your-embedding-deployment",
    openai_api_version="2024-02-01",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=self.api_key
)
```

---

#### Ubicación: Línea 220 (_setup_chain)

**Actual (Ollama)**:
```python
llm = ChatOllama(model=self.llm_model)
```

**Para Gemini**:
```python
llm = ChatGoogleGenerativeAI(
    model=self.llm_model,
    google_api_key=self.api_key,
    temperature=0.7,  # Opcional
    convert_system_message_to_human=True  # Para compatibilidad
)
```

**Para OpenAI**:
```python
llm = ChatOpenAI(
    model=self.llm_model,
    openai_api_key=self.api_key,
    temperature=0.7
)
```

**Para Azure OpenAI**:
```python
llm = AzureChatOpenAI(
    azure_deployment="your-chat-deployment",
    openai_api_version="2024-02-01",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=self.api_key,
    temperature=0.7
)
```

---

### 3. Cambios en `multi_rag_engine.py`

#### Ubicación: Líneas 27-38

**Actual**:
```python
self.rag_distribuidor = RAGEngine(
    data_dir="dataSet",
    persist_dir="./chroma_distribuidor",
    recursive=True
)

self.rag_cliente = RAGEngine(
    data_dir="dataSet/datasetGen",
    persist_dir="./chroma_cliente",
    recursive=False
)
```

**Para Gemini**:
```python
api_key = os.environ.get("GOOGLE_API_KEY")

self.rag_distribuidor = RAGEngine(
    data_dir="dataSet",
    persist_dir="./chroma_distribuidor",
    recursive=True,
    embedding_model="models/embedding-001",
    llm_model="gemini-1.5-pro",
    api_key=api_key
)

self.rag_cliente = RAGEngine(
    data_dir="dataSet/datasetGen",
    persist_dir="./chroma_cliente",
    recursive=False,
    embedding_model="models/embedding-001",
    llm_model="gemini-1.5-pro",
    api_key=api_key
)
```

---

### 4. Variables de Entorno

Crear archivo `.env`:

**Para Gemini**:
```bash
GOOGLE_API_KEY=tu_api_key_aqui
```

**Para OpenAI**:
```bash
OPENAI_API_KEY=tu_api_key_aqui
```

**Para Azure OpenAI**:
```bash
AZURE_OPENAI_API_KEY=tu_api_key_aqui
AZURE_OPENAI_ENDPOINT=https://tu-recurso.openai.azure.com/
```

Cargar en `server/run.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

### 5. Consideraciones de Costos

| Proveedor | Embeddings | LLM | Notas |
|-----------|-----------|-----|-------|
| **Ollama** | Gratis (local) | Gratis (local) | Requiere GPU/CPU potente |
| **Gemini** | Gratis hasta límite | Gratis hasta límite | Gemini 1.5 Flash es más barato |
| **OpenAI** | $0.0001/1K tokens | $0.01-0.03/1K tokens | GPT-4 es caro |
| **Azure OpenAI** | Similar OpenAI | Similar OpenAI | Requiere suscripción Azure |

⚠️ **Importante**: Al usar APIs pagas:
- Implementa **rate limiting**
- Agrega **caching** de respuestas
- Monitorea **costos**

---

### 6. Optimizaciones Recomendadas

#### Para APIs Pagas (Gemini, OpenAI)

**a) Implementar caché de respuestas**:
```python
# En rag_engine.py
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

**b) Reducir chunks para menos embeddings**:
```python
# Línea 169 en rag_engine.py
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Aumentar de 1000 a 1500
    chunk_overlap=150   # Ajustar proporcionalmente
)
```

**c) Limitar resultados del retriever**:
```python
# Línea 240 en rag_engine.py
retriever = MultiQueryRetriever.from_llm(
    self.vector_db.as_retriever(search_kwargs={"k": 3}),  # Solo top 3
    llm,
    prompt=QUERY_PROMPT,
    parser_key="lines"
)
```

---

### 7. Comparación de Modelos

#### Embeddings

| Modelo | Dimensiones | Contexto | Proveedor |
|--------|------------|----------|-----------|
| qwen3-embedding:8b | 1024 | 8192 | Ollama (local) |
| text-embedding-3-large | 3072 | 8191 | OpenAI |
| text-embedding-ada-002 | 1536 | 8191 | OpenAI |
| models/embedding-001 | 768 | 2048 | Google Gemini |

#### LLMs

| Modelo | Contexto | Velocidad | Calidad | Proveedor |
|--------|----------|-----------|---------|-----------|
| llama3.1:latest | 128K | Media | Buena | Ollama (local) |
| gpt-4-turbo | 128K | Lenta | Excelente | OpenAI |
| gpt-3.5-turbo | 16K | Rápida | Buena | OpenAI |
| gemini-1.5-pro | 1M | Media | Excelente | Google |
| gemini-1.5-flash | 1M | Rápida | Muy buena | Google |

---

### 8. Testing Después de la Migración

```python
# test_migration.py
from multi_rag_engine import MultiRAGEngine

def test_basic_query():
    rag = MultiRAGEngine()
    
    # Test 1: Cliente
    result_cliente = rag.ask("Test question", db_type="cliente")
    assert len(result_cliente) > 0, "No response from cliente"
    
    # Test 2: Distribuidor
    result_dist = rag.ask("Test question", db_type="distribuidor")
    assert len(result_dist) > 0, "No response from distribuidor"
    
    print("✅ Migración exitosa!")

if __name__ == "__main__":
    test_basic_query()
```

---

## Resumen de Cambios para Migración

### Checklist

- [ ] Actualizar `requirements.txt` con dependencias de nueva API
- [ ] Agregar imports en `rag_engine.py`
- [ ] Actualizar constructor `__init__` para aceptar `api_key`
- [ ] Cambiar `OllamaEmbeddings` → Nueva clase de embeddings
- [ ] Cambiar `ChatOllama` → Nueva clase de LLM
- [ ] Actualizar `multi_rag_engine.py` para pasar `api_key`
- [ ] Configurar variables de entorno (`.env`)
- [ ] Instalar `python-dotenv` y cargar en `run.py`
- [ ] Testear con consultas básicas
- [ ] Monitorear costos (si aplica)
- [ ] Implementar caché (opcional pero recomendado)
- [ ] Ajustar parámetros (chunk_size, k, temperature)

---

## Notas Finales

- **Performance**: Ollama (local) es más lento pero gratis. APIs en la nube son más rápidas.
- **Privacidad**: Ollama mantiene datos locales. APIs envían datos a terceros.
- **Escalabilidad**: APIs en la nube escalan mejor para múltiples usuarios.
- **Costos**: Ollama requiere hardware. APIs tienen costos por uso.

Para proyectos sensibles o con alta demanda, considera usar **Ollama en producción** con GPUs dedicadas o servicios como **vLLM** para auto-hosting optimizado.
