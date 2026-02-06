"""
Motor Multi-RAG (Orquestador)

Este módulo gestiona múltiples instancias de RAGEngine para soportar
diferentes contextos de conocimiento (ej: cliente vs distribuidor).

Flujo:
1. Inicializa dos bases de datos vectoriales separadas:
   - "cliente": Incluye TODO el contenido (dataSet/ + dataSet/datasetGen/)
   - "distribuidor": Incluye SOLO contenido restringido (dataSet/datasetGen/)

2. Enruta las consultas a la instancia correcta según el parámetro db_type.
"""

from rag_engine import RAGEngine

class MultiRAGEngine:
    def __init__(self):
        """
        Inicializa las dos instancias de RAGEngine.
	Una instancia es para que tenga accceso a los subdir, donde recursive = true
	Y la otra para que solo sea de esa carpeta
        """
        print("=== Inicializando Multi-RAG System ===")
        
        print("\n--- Cargando RAG Distribuidor ---")
        self.rag_distribuidor = RAGEngine(
            data_dir="dataSet",
            persist_dir="./chroma_distribuidor",
            recursive=True  
        )
        
        print("\n--- Cargando RAG Cliente ---")
        self.rag_cliente = RAGEngine(
            data_dir="dataSet/datasetGen",
            persist_dir="./chroma_cliente",
            recursive=False  # Solo archivos en ese directorio
        )
        

    
    def get_engine(self, db_type="cliente"):
        """
        Retorna la instancia de RAGEngine correspondiente.
        
        Args:
            db_type: "cliente" o "distribuidor"
        """
        if db_type == "distribuidor":
            return self.rag_distribuidor
        elif db_type == "cliente":
            return self.rag_cliente
        else:
            # cliente por defecto
            print(f"Advertencia: Tipo de DB '{db_type}'. Usando 'cliente'.")
            return self.rag_cliente

    def ask(self, question, db_type="cliente"):
        """
        Enruta la pregunta al motor correspondiente.
        """
        engine = self.get_engine(db_type)
        #print(f"Bd: {db_type}")
        return engine.ask(question)

    def update_database(self):
        """
        Actualiza AMBAS bases de datos.
        
        Proceso:
        1. Libera todos los recursos (vector_db, chain)
        2. Espera a que el SO libere los locks
        3. Elimina directorios
        4. Recrea las instancias desde cero
        """
        import gc
        import time
        import shutil
        import os
        
        print("\n=== Actualizando TODAS las bases de datos ===")
        
        # PASO 1: Liberar todos los recursos de ambas instancias
        print("Liberando recursos...")
        
        # Distribuidor
        if hasattr(self.rag_distribuidor, 'vector_db'):
            self.rag_distribuidor.vector_db = None
        if hasattr(self.rag_distribuidor, 'chain'):
            self.rag_distribuidor.chain = None
        
        # Cliente
        if hasattr(self.rag_cliente, 'vector_db'):
            self.rag_cliente.vector_db = None
        if hasattr(self.rag_cliente, 'chain'):
            self.rag_cliente.chain = None
        
        gc.collect()
        time.sleep(1.0)
        
        # PASO 2: Eliminar directorios
        print("Eliminando directorios antiguos...")
        for persist_dir in ["./chroma_distribuidor", "./chroma_cliente"]:
            if os.path.exists(persist_dir):
                try:
                    shutil.rmtree(persist_dir)
                    print(f"  ✓ {persist_dir} eliminado")
                except Exception as e:
                    print(f"  ✗ Error eliminando {persist_dir}: {e}")
        
        time.sleep(1.0)
        
        # PASO 3: Recrear las instancias completamente
        print("\nRecreando instancias...")
        
        print("  Distribuidor...")
        self.rag_distribuidor = RAGEngine(
            data_dir="dataSet",
            persist_dir="./chroma_distribuidor",
            recursive=True
        )
        
        print("  Cliente...")
        self.rag_cliente = RAGEngine(
            data_dir="dataSet/datasetGen",
            persist_dir="./chroma_cliente",
            recursive=False
        )
        
        print("\n=== Actualización Completa ✓ ===")
