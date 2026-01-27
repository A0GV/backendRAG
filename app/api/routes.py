"""
Rutas de la API REST para el sistema RAG.
"""
from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    IngestRequest,
    IngestResponse,
    DocumentsResponse,
    DocumentInfo,
    HealthResponse
)
from app.services.document_loader import get_document_loader
from app.services.vector_store import get_vector_store_service
from app.services.llm_service import get_llm_service
from app.config import get_settings


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Verifica el estado del sistema y sus dependencias.
    """
    settings = get_settings()
    llm_service = get_llm_service()
    vector_store = get_vector_store_service()
    
    ollama_ok = llm_service.check_ollama_connection_sync()
    chroma_ok = vector_store.is_initialized()
    openai_ok = bool(settings.openai_api_key and settings.openai_api_key != "sk-your-openai-api-key-here")
    
    status_msg = "healthy" if (ollama_ok and chroma_ok and openai_ok) else "degraded"
    
    return HealthResponse(
        status=status_msg,
        ollama_connected=ollama_ok,
        chroma_initialized=chroma_ok,
        openai_configured=openai_ok
    )


@router.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_documents(request: IngestRequest = None):
    """
    Ingesta documentos Markdown al vector store.
    
    - Carga todos los archivos .md del directorio especificado
    - Los divide en chunks
    - Genera embeddings con OpenAI
    - Los almacena en ChromaDB
    """
    try:
        document_loader = get_document_loader()
        vector_store = get_vector_store_service()
        
        directory = request.directory if request else None
        
        # Cargar y dividir documentos
        chunks, files = document_loader.load_and_split(directory)
        
        if not chunks:
            return IngestResponse(
                status="warning",
                documents_processed=0,
                chunks_created=0,
                files=[]
            )
        
        # Añadir al vector store
        vector_store.add_documents(chunks)
        
        return IngestResponse(
            status="success",
            documents_processed=len(files),
            chunks_created=len(chunks),
            files=files
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la ingesta: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Realiza una consulta al sistema RAG.
    
    - Busca documentos relevantes en ChromaDB
    - Genera una respuesta contextualizada con Ollama
    - Devuelve la respuesta con las fuentes utilizadas
    """
    try:
        llm_service = get_llm_service()
        
        result = llm_service.query(
            question=request.question,
            k=request.top_k
        )
        
        sources = [
            SourceDocument(
                content=src["content"],
                source=src["source"],
                score=src["score"]
            )
            for src in result["sources"]
        ]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            model=result["model"],
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando la consulta: {str(e)}"
        )


@router.get("/documents", response_model=DocumentsResponse, tags=["Documents"])
async def list_documents():
    """
    Lista todos los documentos indexados en el sistema.
    """
    try:
        vector_store = get_vector_store_service()
        stats = vector_store.get_document_stats()
        
        documents = [
            DocumentInfo(
                filename=doc["filename"],
                chunk_count=doc["chunk_count"]
            )
            for doc in stats["documents"]
        ]
        
        return DocumentsResponse(
            total_documents=stats["total_documents"],
            total_chunks=stats["total_chunks"],
            documents=documents
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo documentos: {str(e)}"
        )


@router.delete("/documents", tags=["Documents"])
async def clear_documents():
    """
    Elimina todos los documentos del vector store.
    """
    try:
        vector_store = get_vector_store_service()
        vector_store.clear_collection()
        
        return {"status": "success", "message": "Todos los documentos han sido eliminados"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error eliminando documentos: {str(e)}"
        )
