"""
Backend RAG con FastAPI, Ollama y ChromaDB.

Sistema de Retrieval-Augmented Generation para consultas sobre documentos Markdown.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle del servidor - inicialización y limpieza."""
    # Startup
    settings = get_settings()
    print(f"🚀 Iniciando Backend RAG")
    print(f"📁 Knowledge directory: {settings.knowledge_dir}")
    print(f"🤖 Ollama model: {settings.ollama_model}")
    print(f"💾 ChromaDB: {settings.chroma_persist_directory}")
    yield
    # Shutdown
    print("👋 Cerrando Backend RAG")


app = FastAPI(
    title="Backend RAG - Markdown Knowledge Base",
    description="""
## Sistema RAG con Ollama y ChromaDB

API para consultar una base de conocimiento construida a partir de archivos Markdown.

### Características:
- 📚 **Ingesta de documentos**: Procesa archivos `.md` y los indexa
- 🔍 **Búsqueda semántica**: Encuentra información relevante usando embeddings
- 🤖 **Generación de respuestas**: Usa Ollama para generar respuestas contextualizadas
- 💾 **Persistencia**: ChromaDB mantiene los datos entre reinicios

### Flujo de uso:
1. Coloca tus archivos `.md` en la carpeta `knowledge/`
2. Llama a `POST /api/ingest` para indexar los documentos
3. Usa `POST /api/query` para hacer preguntas
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS - permitir todos los orígenes para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(router, prefix="/api")


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz con información del servicio."""
    return {
        "service": "Backend RAG - Markdown Knowledge Base",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "ingest": "/api/ingest",
            "query": "/api/query",
            "documents": "/api/documents"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
