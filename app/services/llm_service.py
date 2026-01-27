"""
Servicio LLM con Ollama para RAG.
"""
import time
from typing import Optional, List
import httpx

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from app.config import get_settings
from app.services.vector_store import get_vector_store_service


# Template para RAG en español
RAG_PROMPT_TEMPLATE = """Eres un asistente experto que responde preguntas basándose únicamente en el contexto proporcionado.

CONTEXTO:
{context}

INSTRUCCIONES:
1. Responde SOLO con información del contexto proporcionado.
2. Si la información no está en el contexto, di claramente "No tengo información sobre esto en los documentos disponibles."
3. Sé conciso pero completo.
4. Si es relevante, menciona de qué documento proviene la información.

PREGUNTA: {question}

RESPUESTA:"""


class LLMService:
    """Servicio para interactuar con Ollama y realizar RAG."""
    
    def __init__(self):
        self.settings = get_settings()
        self._llm: Optional[ChatOllama] = None
        self._chain = None
        self.vector_store = get_vector_store_service()
    
    @property
    def llm(self) -> ChatOllama:
        """Obtiene el modelo LLM de Ollama."""
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.settings.ollama_model,
                base_url=self.settings.ollama_base_url,
                temperature=0.7,
            )
        return self._llm
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Formatea los documentos recuperados para el prompt."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "Desconocido")
            formatted.append(f"[Documento {i} - {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)
    
    def create_rag_chain(self, k: Optional[int] = None):
        """
        Crea la cadena RAG con retrieval y generación.
        
        Args:
            k: Número de documentos a recuperar
        """
        retriever = self.vector_store.get_retriever(k=k)
        
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(
        self,
        question: str,
        k: Optional[int] = None
    ) -> dict:
        """
        Realiza una consulta RAG.
        
        Args:
            question: Pregunta del usuario
            k: Número de documentos a recuperar
            
        Returns:
            Diccionario con la respuesta y metadatos
        """
        start_time = time.time()
        
        # Obtener documentos relevantes con scores
        docs_with_scores = self.vector_store.similarity_search(question, k=k)
        
        # Extraer solo los documentos para el chain
        docs = [doc for doc, _ in docs_with_scores]
        
        # Crear prompt con contexto
        context = self._format_docs(docs)
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        # Generar respuesta
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        processing_time = time.time() - start_time
        
        # Preparar sources
        sources = [
            {
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "source": doc.metadata.get("filename", "Desconocido"),
                "score": float(score)
            }
            for doc, score in docs_with_scores
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "model": self.settings.ollama_model,
            "processing_time": round(processing_time, 2)
        }
    
    async def check_ollama_connection(self) -> bool:
        """Verifica la conexión con Ollama."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.settings.ollama_base_url}/api/tags",
                    timeout=5.0
                )
                return response.status_code == 200
        except Exception:
            return False
    
    def check_ollama_connection_sync(self) -> bool:
        """Verifica la conexión con Ollama (síncrono)."""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.settings.ollama_base_url}/api/tags",
                    timeout=5.0
                )
                return response.status_code == 200
        except Exception:
            return False


# Singleton
_llm_service = None

def get_llm_service() -> LLMService:
    """Obtiene la instancia singleton del LLMService."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
