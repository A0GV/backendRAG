# Sistema RAG con OpenRouter/OpenAI

Sistema de Retrieval-Augmented Generation (RAG) que responde preguntas basándose en documentos locales (.txt y .md) usando OpenAI vía OpenRouter. Soporta dos bases de conocimiento independientes con acceso diferenciado por rol.

## Requisitos

- Python 3.10+
- Cuenta y API key en [OpenRouter](https://openrouter.ai)

## Instalación

```bash
git clone <repo>
cd backendRAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env y agregar: OPENROUTER_API_KEY=sk-or-v1-...
```

Para obtener la API key: https://openrouter.ai/keys

## Estructura del proyecto

```
backendRAG/
├── rag_engine.py           # Motor RAG (clase RAGEngine)
├── multi_rag_engine.py     # Orquestador dual (clase MultiRAGEngine)
├── tre.py                  # CLI interactivo
├── server/
│   ├── run.py
│   └── app/
│       ├── __init__.py
│       └── routes.py       # Endpoints Flask
├── dataSet/                # Documentos solo para distribuidor
│   └── datasetGen/         # Documentos para cliente y distribuidor
├── chroma_distribuidor/    # ChromaDB distribuidor (generada automaticamente)
├── chroma_cliente/         # ChromaDB cliente (generada automaticamente)
├── .env                    # API key (no subir a git)
└── .env.example            # Plantilla de configuracion
```

## Arquitectura

El sistema tiene dos niveles:

**RAGEngine** — motor individual para una base de datos ChromaDB:
1. Carga archivos `.txt` y `.md` desde un directorio
2. Divide el texto en chunks (1000 chars, overlap 100)
3. Genera embeddings con `text-embedding-3-small` via OpenRouter
4. Almacena en ChromaDB (persiste en disco)
5. Responde preguntas usando MultiQueryRetriever + GPT-4o

**MultiRAGEngine** — orquestador de dos instancias RAGEngine:
- `distribuidor`: accede a todo `dataSet/` (incluye subdirectorios)
- `cliente`: accede solo a `dataSet/datasetGen/`
- Enruta las consultas y operaciones al motor correcto
- Sincroniza: archivos subidos/borrados de cliente tambien se aplican en distribuidor

Flujo de una consulta:
1. El cliente envia pregunta + `db_type`
2. MultiRAGEngine selecciona el RAGEngine correcto
3. RAGEngine genera 5 variaciones de la pregunta (MultiQueryRetriever)
4. Busca chunks similares en ChromaDB
5. GPT-4o genera respuesta basada en el contexto recuperado

## Bases de conocimiento

| Rol | Datos accesibles | Directorio fisico |
|-----|-----------------|-------------------|
| `cliente` | Solo `datasetGen/` | `dataSet/datasetGen/` |
| `distribuidor` | Todo `dataSet/` | `dataSet/` (recursivo) |

Un archivo subido con `target=cliente` se guarda en `datasetGen/` y se indexa en ambas bases.
Un archivo subido con `target=distribuidor` se guarda en `dataSet/` y solo se indexa en distribuidor.

## Uso: CLI interactivo

```bash
python3 tre.py
```

Comandos disponibles en el CLI:

| Comando | Descripcion |
|---------|-------------|
| `use cliente` | Cambia a base de datos cliente |
| `use distribuidor` | Cambia a base de datos distribuidor |
| `update` | Reconstruye ambas bases de datos desde disco |
| `q` | Salir |

## Uso: API REST

Iniciar el servidor:

```bash
cd server
python3 run.py
```

El servidor queda disponible en `http://localhost:5000`.

---

### GET /

Health check. Retorna `{"status": "ok"}` si el sistema esta listo.

---

### POST /api/ask

Hace una pregunta al sistema RAG.

Request:
```json
{
  "question": "Que son las geocercas?",
  "db_type": "distribuidor"
}
```

`db_type` es opcional, por defecto `"cliente"`.

Response:
```json
{
  "success": true,
  "answer": "Las geocercas son perimetros virtuales...",
  "db_used": "distribuidor"
}
```

Ejemplo con curl:
```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Que son las geocercas?", "db_type": "distribuidor"}'
```

Ejemplo con JavaScript:
```javascript
const response = await fetch('http://localhost:5000/api/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: 'Que son las geocercas?', db_type: 'distribuidor' })
});
const data = await response.json();
console.log(data.answer);
```

---

### POST /api/update

Reconstruye ambas bases de datos desde los archivos en disco. Util despues de editar o eliminar archivos manualmente.

Request: sin body.

Response:
```json
{
  "success": true,
  "message": "All databases updated successfully"
}
```

---

### POST /api/upload

Sube un archivo nuevo y lo indexa incrementalmente sin reconstruir la base de datos.

Request (multipart/form-data):

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `file` | File | Archivo `.txt` o `.md` |
| `target` | Text | `"cliente"` o `"distribuidor"` |

Response:
```json
{
  "success": true,
  "message": "File uploaded and indexed successfully",
  "filename": "manual.md",
  "target": "cliente",
  "chunks_added": 15
}
```

Ejemplo con curl:
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@/ruta/al/archivo.md" \
  -F "target=cliente"
```

Ejemplo con JavaScript:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('target', 'cliente');

const response = await fetch('http://localhost:5000/api/upload', {
  method: 'POST',
  body: formData
});
const data = await response.json();
```

Ejemplo con Python:
```python
import requests

with open('archivo.md', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/upload',
        files={'file': f},
        data={'target': 'distribuidor'}
    )
print(response.json())
```

---

### POST /api/delete

Elimina un documento de la base de datos vectorial y del disco.

Request (JSON):

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `filename` | String | Nombre del archivo (ej: `"manual.md"`) |
| `target` | String | `"cliente"` o `"distribuidor"` |

Response:
```json
{
  "success": true,
  "message": "File deleted successfully",
  "filename": "manual.md",
  "target": "cliente"
}
```

Ejemplo con curl:
```bash
curl -X POST http://localhost:5000/api/delete \
  -H "Content-Type: application/json" \
  -d '{"filename": "manual_obsoleto.md", "target": "cliente"}'
```

Nota: si el target es `cliente`, el archivo tambien se elimina del indice de distribuidor.

---

## Costos (OpenRouter)

Modelos en uso:
- GPT-4o: ~$2.50/1M tokens entrada, ~$10/1M tokens salida
- text-embedding-3-small: ~$0.02/1M tokens

Estimaciones:
- Indexar 50 documentos (~50k tokens): ~$0.001
- Consulta tipica (1k entrada + 500 salida): ~$0.008
- 1000 consultas/mes: ~$8

Monitorea el uso en https://openrouter.ai/activity

## Seguridad

- El archivo `.env` esta en `.gitignore` y no se sube a git
- Nunca hagas commit de la API key
- La API key se carga desde variables de entorno con `python-dotenv`

## Desarrollo

### Agregar documentos manualmente

1. Copiar archivos `.txt` o `.md` a `dataSet/` (distribuidor) o `dataSet/datasetGen/` (cliente)
2. Ejecutar `update` en el CLI o llamar `POST /api/update`
3. Las ChromaDBs se reconstruyen con todos los archivos del directorio

### Cambiar modelos

Editar los parametros en `multi_rag_engine.py`:

```python
self.rag_distribuidor = RAGEngine(
    llm_model="openai/gpt-3.5-turbo",       # modelo mas barato
    embedding_model="text-embedding-3-small"
)
```

Lista de modelos disponibles: https://openrouter.ai/models

## Troubleshooting

**"OPENROUTER_API_KEY no encontrada"**
Verificar que el archivo `.env` existe en la raiz del proyecto y contiene:
```
OPENROUTER_API_KEY=sk-or-v1-tu-key-aqui
```

**"attempt to write a readonly database"**
ChromaDB tiene locks activos. Eliminar los directorios y reiniciar:
```bash
rm -rf chroma_cliente chroma_distribuidor
python3 tre.py
```

**Rate limit / error 429**
OpenRouter reintenta automaticamente. Si persiste, revisar limites en https://openrouter.ai/settings/limits

**El modelo sigue respondiendo sobre contenido eliminado**
Al editar o borrar archivos en disco, los vectores en ChromaDB no se actualizan automaticamente. Usar `update` (CLI) o `POST /api/update` para reconstruir la base, o usar `POST /api/delete` para borrar un archivo especifico.

## Licencia

MIT
