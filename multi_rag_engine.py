from rag_engine import RAGEngine

class MultiRAGEngine:
    def __init__(self):
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
            recursive=False
        )

    def get_engine(self, db_type="cliente"):
        if db_type == "distribuidor":
            return self.rag_distribuidor
        elif db_type == "cliente":
            return self.rag_cliente
        else:
            print(f"Advertencia: Tipo de DB '{db_type}'. Usando 'cliente'.")
            return self.rag_cliente

    def ask(self, question, db_type="cliente"):
        engine = self.get_engine(db_type)
        return engine.ask(question)

    def update_database(self):
        import gc
        import time
        import shutil
        import os

        print("\n=== Actualizando TODAS las bases de datos ===")

        print("Liberando recursos...")
        if hasattr(self.rag_distribuidor, 'vector_db'):
            self.rag_distribuidor.vector_db = None
        if hasattr(self.rag_distribuidor, 'chain'):
            self.rag_distribuidor.chain = None

        if hasattr(self.rag_cliente, 'vector_db'):
            self.rag_cliente.vector_db = None
        if hasattr(self.rag_cliente, 'chain'):
            self.rag_cliente.chain = None

        gc.collect()
        time.sleep(1.0)

        print("Eliminando directorios antiguos...")
        for persist_dir in ["./chroma_distribuidor", "./chroma_cliente"]:
            if os.path.exists(persist_dir):
                try:
                    shutil.rmtree(persist_dir)
                    print(f"  ✓ {persist_dir} eliminado")
                except Exception as e:
                    print(f"  ✗ Error eliminando {persist_dir}: {e}")

        time.sleep(1.0)

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
        print(f"\n=== Agregando documento incremental ===")
        print(f"Archivo: {file_path}")
        print(f"Target: {target}")

        target = target.lower().strip()
        chunks_added = 0

        if target == "distribuidor":
            chunks_added = self.rag_distribuidor.add_document(file_path)

        elif target == "cliente":
            chunks_added = self.rag_cliente.add_document(file_path)

            # Los archivos de cliente tambien se indexan en distribuidor
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
        print(f"\n=== Eliminando documento incremental ===")
        print(f"Archivo: {filename}")
        print(f"Target: {target}")

        target = target.lower().strip()
        import os

        if target == "distribuidor":
            file_dir = "dataSet"
        elif target == "cliente":
            file_dir = "dataSet/datasetGen"
        else:
            raise ValueError(f"Target inválido: {target}. Use 'distribuidor' o 'cliente'.")

        file_path = os.path.join(file_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {filename} no existe en {file_dir}")

        deleted_count = 0

        if target == "distribuidor":
            deleted = self.rag_distribuidor.delete_document(file_path)
            if deleted: deleted_count += 1

        elif target == "cliente":
            deleted_cli = self.rag_cliente.delete_document(file_path)
            if deleted_cli: deleted_count += 1

            # Sincronizar eliminacion con distribuidor
            print(">> Sincronizando eliminación con Distribuidor (acceso total)...")
            try:
                deleted_dist = self.rag_distribuidor.delete_document(file_path)
                if deleted_dist: deleted_count += 1
                print(">> Distribuidor sincronizado correctamente.")
            except Exception as e:
                print(f"Warning: No se pudo sincronizar con Distribuidor: {e}")

        try:
            os.remove(file_path)
            print(f"✓ Archivo {file_path} eliminado del disco.")
        except Exception as e:
            print(f"Error eliminando archivo del disco: {e}")
            raise e

        print(f"=== Documento eliminado exitosamente ===\n")
        return {"target": target, "filename": filename, "deleted": True}
