"""
Configuración centralizada del proyecto usando Pydantic Settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Configuración del backend RAG."""
    
    # OpenAI (para embeddings)
    openai_api_key: str = Field(..., description="OpenAI API Key para embeddings")
    
    # Ollama (LLM local)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="URL base de Ollama"
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Modelo de Ollama a usar"
    )
    
    # ChromaDB
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        description="Directorio de persistencia de ChromaDB"
    )
    chroma_collection_name: str = Field(
        default="markdown_docs",
        description="Nombre de la colección en ChromaDB"
    )
    
    # Knowledge Base
    knowledge_dir: str = Field(
        default="./knowledge",
        description="Directorio con archivos .md"
    )
    
    # RAG Settings
    chunk_size: int = Field(
        default=1000,
        description="Tamaño de los chunks de texto"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Solapamiento entre chunks"
    )
    top_k_results: int = Field(
        default=4,
        description="Número de documentos a recuperar"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Obtiene la configuración cacheada."""
    return Settings()
