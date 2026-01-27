"""
Servicio para gestionar el vector store con ChromaDB y OpenAI embeddings.
"""
from typing import List, Optional
from collections import defaultdict

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.config import get_settings


class VectorStoreService:
    """Gestiona el vector store ChromaDB con OpenAI embeddings."""
    
    def __init__(self):
        self.settings = get_settings()
        self._vectorstore: Optional[Chroma] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._client: Optional[chromadb.PersistentClient] = None
    
    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Obtiene el modelo de embeddings de OpenAI."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                openai_api_key=self.settings.openai_api_key,
                model="text-embedding-3-small"  # Más económico y eficiente
            )
        return self._embeddings
    
    @property
    def client(self) -> chromadb.PersistentClient:
        """Obtiene el cliente de ChromaDB."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False
                )
            )
        return self._client
    
    @property
    def vectorstore(self) -> Chroma:
        """Obtiene el vector store."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                client=self.client,
                collection_name=self.settings.chroma_collection_name,
                embedding_function=self.embeddings
            )
        return self._vectorstore
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Añade documentos al vector store.
        
        Args:
            documents: Lista de documentos a añadir
            
        Returns:
            Número de documentos añadidos
        """
        if not documents:
            return 0
        
        self.vectorstore.add_documents(documents)
        return len(documents)
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """
        Busca documentos similares a la query.
        
        Args:
            query: Texto de búsqueda
            k: Número de resultados (usa top_k_results por defecto)
            
        Returns:
            Lista de tuplas (documento, score)
        """
        if k is None:
            k = self.settings.top_k_results
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def get_retriever(self, k: Optional[int] = None):
        """
        Obtiene un retriever para usar con LangChain.
        
        Args:
            k: Número de documentos a recuperar
            
        Returns:
            Retriever de LangChain
        """
        if k is None:
            k = self.settings.top_k_results
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def get_document_stats(self) -> dict:
        """
        Obtiene estadísticas de los documentos indexados.
        
        Returns:
            Diccionario con estadísticas
        """
        collection = self.client.get_collection(
            name=self.settings.chroma_collection_name
        )
        
        # Obtener todos los metadatos
        results = collection.get(include=["metadatas"])
        
        total_chunks = len(results["ids"]) if results["ids"] else 0
        
        # Contar chunks por archivo
        file_counts = defaultdict(int)
        for metadata in results["metadatas"] or []:
            filename = metadata.get("filename", "unknown")
            file_counts[filename] += 1
        
        return {
            "total_chunks": total_chunks,
            "total_documents": len(file_counts),
            "documents": [
                {"filename": fname, "chunk_count": count}
                for fname, count in file_counts.items()
            ]
        }
    
    def clear_collection(self):
        """Limpia todos los documentos de la colección."""
        try:
            self.client.delete_collection(
                name=self.settings.chroma_collection_name
            )
            # Recrear vectorstore vacío
            self._vectorstore = None
        except Exception:
            pass  # La colección no existía
    
    def is_initialized(self) -> bool:
        """Verifica si el vector store está inicializado."""
        try:
            _ = self.vectorstore
            return True
        except Exception:
            return False


# Singleton
_vector_store_service = None

def get_vector_store_service() -> VectorStoreService:
    """Obtiene la instancia singleton del VectorStoreService."""
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service
