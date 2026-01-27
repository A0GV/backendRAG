"""
Servicio para cargar y procesar documentos Markdown.
"""
import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import get_settings


class DocumentLoader:
    """Carga y procesa documentos Markdown."""
    
    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=[
                "\n## ",      # Headers nivel 2
                "\n### ",     # Headers nivel 3
                "\n#### ",    # Headers nivel 4
                "\n\n",       # Párrafos
                "\n",         # Líneas
                ". ",         # Oraciones
                " ",          # Palabras
                ""
            ]
        )
    
    def load_markdown_file(self, file_path: str) -> str:
        """
        Carga el contenido de un archivo Markdown.
        
        Args:
            file_path: Ruta al archivo .md
            
        Returns:
            Contenido del archivo como string
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def load_directory(self, directory: str = None) -> List[Document]:
        """
        Carga todos los archivos .md de un directorio.
        
        Args:
            directory: Directorio a procesar (usa knowledge_dir por defecto)
            
        Returns:
            Lista de Documents de LangChain
        """
        if directory is None:
            directory = self.settings.knowledge_dir
        
        documents = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"El directorio no existe: {directory}")
        
        # Buscar archivos .md recursivamente
        for md_file in dir_path.rglob("*.md"):
            content = self.load_markdown_file(str(md_file))
            
            # Crear documento con metadata
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file),
                    "filename": md_file.name,
                    "file_path": str(md_file.relative_to(dir_path))
                }
            )
            documents.append(doc)
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide los documentos en chunks más pequeños.
        
        Args:
            documents: Lista de documentos a dividir
            
        Returns:
            Lista de chunks como Documents
        """
        return self.text_splitter.split_documents(documents)
    
    def load_and_split(self, directory: str = None) -> tuple[List[Document], List[str]]:
        """
        Carga y divide todos los documentos de un directorio.
        
        Args:
            directory: Directorio a procesar
            
        Returns:
            Tupla de (chunks, lista de archivos procesados)
        """
        documents = self.load_directory(directory)
        
        if not documents:
            return [], []
        
        chunks = self.split_documents(documents)
        files_processed = list(set(doc.metadata["filename"] for doc in documents))
        
        return chunks, files_processed


# Singleton
_document_loader = None

def get_document_loader() -> DocumentLoader:
    """Obtiene la instancia singleton del DocumentLoader."""
    global _document_loader
    if _document_loader is None:
        _document_loader = DocumentLoader()
    return _document_loader
