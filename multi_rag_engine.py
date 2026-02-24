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

    def add_document_to_target(self, file_path, target):
        """
        Agrega un documento a la base de datos del target especificado.
        
        Args:
            file_path: Ruta al archivo .txt/.md
            target: "distribuidor" o "cliente"
        
        Returns:
            dict: Stats de la operación
        """
        print(f"\n=== Agregando documento incremental ===")
        print(f"Archivo: {file_path}")
        print(f"Target: {target}")
        
        target = target.lower().strip()
        chunks_added = 0
        
        if target == "distribuidor":
            # Distribuidor tiene acceso a todo dataSet/
            chunks_added = self.rag_distribuidor.add_document(file_path)
            
        elif target == "cliente":
            # Cliente tiene acceso solo a datasetGen/
            # Se indexa en Cliente
            chunks_added = self.rag_cliente.add_document(file_path)
            
            # Sincronizar con Distribuidor (que tiene acceso total)
            print(">> Sincronizando también con Distribuidor (acceso total)...")
            try:
                self.rag_distribuidor.add_document(file_path)
                print(">> Distribuidor sincronizado correctamente.")
            except Exception as e:
                print(f"Warning: No se pudo sincronizar con Distribuidor: {e}")
            
        else:
            raise ValueError(f"Target inválido: {target}. Use 'distribuidor' o 'cliente'.")
            
        print(f"=== Documento agregado exitosamente ({chunks_added} chunks) ===\n")
        return {"target": target, "chunks_added": chunks_added}
        
    def delete_document_from_target(self, filename, target):
        """
        Elimina un documento de la base de datos y del disco.
        
        Args:
            filename: Nombre del archivo (ej: "manual.md")
            target: "distribuidor" o "cliente"
            
        Returns:
            dict: Stats de la operación
        """
        print(f"\n=== Eliminando documento incremental ===")
        print(f"Archivo: {filename}")
        print(f"Target: {target}")
        
        target = target.lower().strip()
        import os
        
        # 1. Determinar path y directorio
        if target == "distribuidor":
            file_dir = "dataSet"
        elif target == "cliente":
            file_dir = "dataSet/datasetGen"
        else:
            raise ValueError(f"Target inválido: {target}. Use 'distribuidor' o 'cliente'.")
            
        file_path = os.path.join(file_dir, filename)
        
        # 2. Verificar existencia en disco
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {filename} no existe en {file_dir}")
            
        # 3. Eliminar de las bases de datos vectoriales
        # IMPORTANTE: Debemos usar el PATH RELATIVO o ABSOLUTO EXACTO que se usó al cargar.
        # TextLoader y nuestro sistema usan paths relativos al cargar.
        
        deleted_count = 0
        
        if target == "distribuidor":
            # Distribuidor tiene acceso a todo dataSet/
            deleted = self.rag_distribuidor.delete_document(file_path)
            if deleted: deleted_count += 1
            
        elif target == "cliente":
            # Cliente tiene acceso solo a datasetGen/
            deleted_cli = self.rag_cliente.delete_document(file_path)
            if deleted_cli: deleted_count += 1
            
            # Sincronizar con Distribuidor (que tiene acceso total)
            print(">> Sincronizando eliminación con Distribuidor (acceso total)...")
            try:
                # El distribuidor ve TODO, así que si borramos de cliente, debemos borrar de distribuidor
                deleted_dist = self.rag_distribuidor.delete_document(file_path)
                if deleted_dist: deleted_count += 1
                print(">> Distribuidor sincronizado correctamente.")
            except Exception as e:
                print(f"Warning: No se pudo sincronizar con Distribuidor: {e}")
        
        # 4. Eliminar del disco
        try:
            os.remove(file_path)
            print(f"✓ Archivo {file_path} eliminado del disco.")
        except Exception as e:
            print(f"Error eliminando archivo del disco: {e}")
            raise e
            
        print(f"=== Documento eliminado exitosamente ===\n")
        return {"target": target, "filename": filename, "deleted": True}
