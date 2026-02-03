import chromadb
from chromadb.utils import embedding_functions

def main():
    print("--- INICIO DEL PROCESO ---")
    
    # 1. Configurar el cliente de ChromaDB
    print("[1/5] Configurando la base de datos ChromaDB...")
    # PersistentClient guardará la base de datos en una carpeta local para que no se pierda al cerrar
    client = chromadb.PersistentClient(path="./chroma_db")

    # 2. Configurar la función de embedding con Ollama
    print("[2/5] Configurando el modelo de embedding 'qwen3-embedding:8b'...")
    # Usamos el modelo específico que solicitaste: 'qwen3-embedding:8b'
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        model_name="qwen3-embedding:8b",
        url="http://localhost:11434/api/embeddings", # Endpoint por defecto de Ollama
    )

    # 3. Crear o conectar a una colección
    print("[3/5] Creando/Conectando a la colección 'cat_knowledge_base'...")
    collection = client.get_or_create_collection(
        name="cat_knowledge_base",
        embedding_function=ollama_ef
    )

    # 4. Leer el archivo de datos
    file_path = 'cat-facts.txt'
    print(f"[4/5] Leyendo el archivo '{file_path}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Filtramos líneas vacías
            documents = [line.strip() for line in file.readlines() if line.strip()]
        print(f"      -> Se cargaron {len(documents)} hechos sobre gatos.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {file_path}")
        return

    if not documents:
        print("El archivo está vacío.")
        return

    # Generamos IDs simples para cada documento
    ids = [f"id_{i}" for i in range(len(documents))]

    # 5. Insertar (o actualizar) los documentos en ChromaDB
    # Esto llamará automáticamente a Ollama para generar los embeddings
    print(f"[5/5] Generando embeddings (vectores) e indexando en ChromaDB...")
    print(f"      Procesando {len(documents)} documentos (esto puede tardar unos segundos)...")
    collection.upsert(
        documents=documents,
        ids=ids
    )

    print("      -> ¡Indexación completada con éxito!")
    
    # PEQUEÑA PRUEBA DE CONSULTA
    print("\n--- PRUEBA DE RECUPERACIÓN (Retrieval) ---")
    print("NOTA: Este script recupera los fragmentos de texto relevantes (Chunks).")
    print("      No genera una respuesta nueva, solo busca la información exacta en tu archivo.")
    
    query = "How much do cats sleep?"
    print(f"\nPregunta simulada: '{query}'")
    print("Buscando en la base de datos vectorial...")
    
    results = collection.query(
        query_texts=[query],
        n_results=5 # Traemos los 3 más relevantes
    )
    
    print("\nResultados encontrados (Chunks más relevantes):")
    for idx, doc in enumerate(results['documents'][0]):
        print(f"\n[Resultado #{idx+1}]:")
        print(f"\"{doc}\"")

    print("\n--- fin del script ---")

if __name__ == "__main__":
    main()
