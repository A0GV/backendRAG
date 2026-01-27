"""
Schemas Pydantic para request/response de la API.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ==================== Query ====================

class QueryRequest(BaseModel):
    """Request para consultar el sistema RAG."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Pregunta a realizar al sistema RAG"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Número de documentos a recuperar (override del default)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "¿Qué es Python?",
                    "top_k": 4
                }
            ]
        }
    }


class SourceDocument(BaseModel):
    """Documento fuente usado en la respuesta."""
    content: str = Field(..., description="Contenido del chunk")
    source: str = Field(..., description="Archivo fuente")
    score: Optional[float] = Field(None, description="Score de relevancia")


class QueryResponse(BaseModel):
    """Response de una consulta RAG."""
    answer: str = Field(..., description="Respuesta generada por el LLM")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Documentos fuente utilizados"
    )
    model: str = Field(..., description="Modelo LLM usado")
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")


# ==================== Ingest ====================

class IngestRequest(BaseModel):
    """Request para ingestar documentos."""
    directory: Optional[str] = Field(
        default=None,
        description="Directorio a procesar (usa el default si no se especifica)"
    )
    
    
class IngestResponse(BaseModel):
    """Response de la ingesta de documentos."""
    status: str = Field(..., description="Estado de la operación")
    documents_processed: int = Field(..., description="Número de documentos procesados")
    chunks_created: int = Field(..., description="Número de chunks creados")
    files: list[str] = Field(default_factory=list, description="Archivos procesados")


# ==================== Documents ====================

class DocumentInfo(BaseModel):
    """Información de un documento indexado."""
    filename: str = Field(..., description="Nombre del archivo")
    chunk_count: int = Field(..., description="Número de chunks")
    indexed_at: Optional[datetime] = Field(None, description="Fecha de indexación")


class DocumentsResponse(BaseModel):
    """Response con lista de documentos."""
    total_documents: int = Field(..., description="Total de documentos únicos")
    total_chunks: int = Field(..., description="Total de chunks en el sistema")
    documents: list[DocumentInfo] = Field(
        default_factory=list,
        description="Lista de documentos"
    )


# ==================== Health ====================

class HealthResponse(BaseModel):
    """Response del health check."""
    status: str = Field(..., description="Estado del servicio")
    ollama_connected: bool = Field(..., description="Conexión con Ollama")
    chroma_initialized: bool = Field(..., description="ChromaDB inicializado")
    openai_configured: bool = Field(..., description="OpenAI API key configurada")
