# Sistema RAG con OpenRouter/OpenAI

Sistema de Retrieval-Augmented Generation (RAG) que usa OpenAI a través de OpenRouter para responder preguntas basándose en documentos locales.

## 🚀 Características

- **Múltiples bases de datos**: Soporte para diferentes contextos (Cliente/Distribuidor)
- **OpenAI vía OpenRouter**: Usa GPT-4o para respuestas y text-embedding-3-small para vectores
- **Persistencia**: ChromaDB guarda embeddings en disco para reutilización
- **API REST**: Endpoints Flask para integración
- **CLI Interactivo**: Interfaz de línea de comandos con `tre.py`

## 📋 Requisitos

- Python 3.10+
- Cuenta en [OpenRouter](https://openrouter.ai)
- API Key de OpenRouter

## 🔧 Instalación

### 1. Clonar el repositorio

```bash
git clone <tu-repo>
cd backendRAG
```

### 2. Crear entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar API Key

Crea un archivo `.env` en la raíz del proyecto:

```bash
cp .env.example .env
```

Edita `.env` y agrega tu API key de OpenRouter:

```env
OPENROUTER_API_KEY=sk-or-v1-tu-key-aqui
```

**¿Cómo obtener la API key?**
1. Crea una cuenta en https://openrouter.ai
2. Ve a https://openrouter.ai/keys
3. Crea una nueva API key
4. Copia y pega en tu archivo `.env`

## 📁 Estructura de Datos

```
backendRAG/
├── dataSet/              # Documentos para Distribuidor (acceso completo)
│   ├── geocercas.md
│   ├── mapa.md
│   └── datasetGen/       # Documentos para Cliente (acceso restringido)
│       └── info_distribuidor.txt
├── chroma_distribuidor/  # ChromaDB para Distribuidor (generada automáticamente)
├── chroma_cliente/       # ChromaDB para Cliente (generada automáticamente)
└── .env                  # Tu API key (NO SUBIR A GIT)
```

## 🎮 Uso

### Modo CLI Interactivo

```bash
python3 tre.py
```

Comandos disponibles:
- `use cliente` - Cambia a BD restringida (solo datasetGen/)
- `use distribuidor` - Cambia a BD completa (todo dataSet/)
- `update` - Recrea ambas bases de datos
- `q` - Salir

**Ejemplo:**
```
[cliente] Your question: ¿Qué son las geocercas?
# Respuesta: No tengo información (porque está en dataSet/ que Cliente no ve)

[cliente] Your question: use distribuidor
Switched to DISTRIBUIDOR database (Full Access)

[distribuidor] Your question: ¿Qué son las geocercas?
# Respuesta: Las geocercas son perímetros virtuales...
```

### Modo API

#### Iniciar servidor

```bash
cd server
python3 run.py
```

El servidor estará disponible en `http://localhost:5000`

#### Endpoints

**POST /api/ask** - Hacer una pregunta

```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Qué son las geocercas?",
    "db_type": "distribuidor"
  }'
```

Respuesta:
```json
{
  "success": true,
  "answer": "Las geocercas son perímetros virtuales...",
  "db_used": "distribuidor"
}
```

**POST /api/update** - Actualizar bases de datos

```bash
curl -X POST http://localhost:5000/api/update
```

### Modo Python

```python
from multi_rag_engine import MultiRAGEngine

# Inicializar
rag = MultiRAGEngine()

# Consulta con distribuidor (acceso completo)
respuesta = rag.ask(
    "¿Qué son las geocercas?", 
    db_type="distribuidor"
)
print(respuesta)

# Consulta con cliente (acceso restringido)
respuesta = rag.ask(
    "¿Cuál es el margen?", 
    db_type="cliente"
)
print(respuesta)

# Actualizar bases de datos
rag.update_database()
```

## 💰 Costos (OpenRouter)

**Modelos usados:**
- **GPT-4o**: ~$2.50/1M tokens input, ~$10/1M tokens output
- **text-embedding-3-small**: ~$0.02/1M tokens

**Estimaciones:**
- Crear embeddings para 50 documentos (~50k tokens): ~$0.001
- Una consulta típica (1k tokens input + 500 tokens output): ~$0.008
- 1000 consultas/mes: ~$8

💡 **Tip**: Monitorea tu uso en el [dashboard de OpenRouter](https://openrouter.ai/activity)

## 🔒 Seguridad

- ✅ `.env` está en `.gitignore` (tu API key NO se sube a Git)
- ✅ Usa variables de entorno para credenciales
- ⚠️ NUNCA hagas commit de tu `.env`
- ⚠️ NUNCA compartas tu API key públicamente

## 🛠️ Desarrollo

### Agregar nuevos documentos

1. Agrega archivos `.txt` o `.md` a `dataSet/` o `dataSet/datasetGen/`
2. Ejecuta `update` en el CLI o llama a `rag.update_database()`
3. Las ChromaDBs se recrearán con los nuevos documentos

### Cambiar modelos

Edita `multi_rag_engine.py`:

```python
# Para usar GPT-3.5-turbo en vez de GPT-4o
self.rag_distribuidor = RAGEngine(
    llm_model="openai/gpt-3.5-turbo",  # Más barato
    embedding_model="text-embedding-3-small"
)
```

Modelos disponibles en OpenRouter: https://openrouter.ai/models

## 📊 Arquitectura

```
┌─────────────────────────────────────┐
│      MultiRAGEngine                 │
│   (Orquestador de contextos)       │
└────────────┬────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼────┐
│RAGEngine│      │RAGEngine│
│Cliente  │      │Distrib. │
│(Restric)│      │(Full)   │
└───┬─────┘      └────┬────┘
    │                 │
    │   OpenAI API    │
    │   via OpenRouter│
    └─────────────────┘
```

### Flujo de una consulta

1. **Usuario**: Envía pregunta + `db_type`
2. **MultiRAGEngine**: Selecciona RAGEngine correcto
3. **RAGEngine**: 
   - Genera 5 variaciones de la pregunta (MultiQueryRetriever)
   - Busca documentos similares en ChromaDB
   - Recupera top chunks relevantes
4. **OpenAI (GPT-4o)**: Genera respuesta basada en contexto
5. **Usuario**: Recibe respuesta

## 🐛 Troubleshooting

### Error: "OPENROUTER_API_KEY no encontrada"

**Solución**: Asegúrate de que el archivo `.env` existe y contiene:
```
OPENROUTER_API_KEY=sk-or-v1-tu-key-aqui
```

### Error: "attempt to write a readonly database"

**Solución**: Elimina las ChromaDBs y vuelve a crearlas:
```bash
rm -rf chroma_cliente chroma_distribuidor
python3 tre.py  # Se recrearán automáticamente
```

### Error: Rate limit / 429

**Solución**: OpenRouter maneja rate limits automáticamente con retries. Si persiste:
1. Verifica tu límite en https://openrouter.ai/settings/limits
2. Espera unos minutos
3. Considera agregar créditos

### Las respuestas están en inglés

**Solución**: El prompt ya incluye "Responde en el mismo idioma de la pregunta". Si persiste, verifica que tu pregunta esté en español.

## 📝 Licencia

MIT

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -am 'Agrega mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o soporte, abre un issue en GitHub.

---

**Nota**: Este proyecto usa OpenRouter para acceder a modelos de OpenAI. OpenRouter es un servicio independiente que facilita el acceso a múltiples proveedores de LLMs con un solo endpoint y API key.
