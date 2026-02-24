# Guía de Integración: Endpoint de Subida de Archivos

Esta guía detalla cómo integrar el endpoint `/api/upload` del sistema RAG en tu aplicación. Este endpoint permite subir nuevos documentos (.txt o .md) y actualiza la base de datos de conocimiento instantáneamente sin detener el servicio.

## 📡 Detalles del Endpoint

- **URL**: `http://localhost:5000/api/upload` (ajusta el host según tu despliegue)
- **Método**: `POST`
- **Content-Type**: `multipart/form-data`

### Parámetros del Formulario (Form Data)

| Campo | Tipo | Obligatorio | Descripción |
|-------|------|-------------|-------------|
| `file` | File | Sí | El archivo a subir. Extensiones permitidas: `.txt`, `.md` |
| `target` | Text | Sí | Destino del conocimiento. Valores: `cliente` o `distribuidor` |

### Comportamiento según `target`

- **`cliente`**: 
  - El archivo se guarda en la carpeta pública.
  - **Visible para**: Clientes Y Distribuidores.
  - **Uso**: Información general, manuales públicos, FAQs.

- **`distribuidor`**: 
  - El archivo se guarda en la carpeta privada.
  - **Visible para**: SOLO Distribuidores.
  - **Uso**: Precios confidenciales, márgenes, guías internas.

---

## 📦 Formato de Respuesta

### ✅ Éxito (200 OK)

```json
{
  "success": true,
  "message": "File uploaded and indexed successfully",
  "filename": "manual_nuevo.md",
  "target": "cliente",
  "chunks_added": 15
}
```

### ❌ Error (400 Bad Request / 500 Server Error)

```json
{
  "success": false,
  "error": "Invalid file extension. Only .txt and .md allowed"
}
```

---

## 💻 Ejemplos de Código

### 1. JavaScript (Frontend / React / Vue)

Usando `fetch` nativo:

```javascript
async function uploadDocument(fileObject, targetRole) {
  const formData = new FormData();
  formData.append('file', fileObject); // fileObject viene de un <input type="file">
  formData.append('target', targetRole); // 'cliente' o 'distribuidor'

  try {
    const response = await fetch('http://localhost:5000/api/upload', {
      method: 'POST',
      body: formData
    });

    const result = await response.json();
    
    if (result.success) {
      console.log('Subida exitosa:', result);
      alert(`Documento agregado. Chunks indexados: ${result.chunks_added}`);
    } else {
      console.error('Error:', result.error);
    }
  } catch (error) {
    console.error('Error de red:', error);
  }
}
```

### 2. Node.js (Axios)

```javascript
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

async function uploadFile() {
  const form = new FormData();
  form.append('file', fs.createReadStream('./documento-confidencial.txt'));
  form.append('target', 'distribuidor');

  try {
    const response = await axios.post('http://localhost:5000/api/upload', form, {
      headers: {
        ...form.getHeaders()
      }
    });
    console.log(response.data);
  } catch (error) {
    console.error(error.response ? error.response.data : error.message);
  }
}

uploadFile();
```

### 3. Python (Requests)

```python
import requests

url = "http://localhost:5000/api/upload"
file_path = "dataset/nuevos_precios.txt"

# Abrir el archivo en modo binario
with open(file_path, "rb") as f:
    files = {"file": f}
    data = {"target": "distribuidor"}
    
    response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    print("Éxito:", response.json())
else:
    print("Error:", response.json())
```

### 4. cURL (Terminal)

```bash
# Subir archivo para clientes
curl -X POST http://localhost:5000/api/upload \
  -F "file=@/ruta/a/mi/archivo.md" \
  -F "target=cliente"

# Subir archivo confidencial para distribuidores
curl -X POST http://localhost:5000/api/upload \
  -F "file=@/ruta/a/secreto.txt" \
  -F "target=distribuidor"
```

---

## 🗑️ Endpoint de Eliminación

Permite eliminar documentos específicos de la base de datos y del sistema de archivos.

### Detalles del Endpoint

- **URL**: `http://localhost:5000/api/delete`
- **Método**: `POST`
- **Content-Type**: `application/json`

### Parámetros (JSON Body)

| Campo | Tipo | Obligatorio | Descripción |
|-------|------|-------------|-------------|
| `filename` | String | Sí | Nombre del archivo a eliminar (ej: "manual.md") |
| `target` | String | Sí | Origen del archivo. Valores: `cliente` o `distribuidor` |

### Comportamiento

- **`cliente`**: Elimina de `dataSet/datasetGen/` y actualiza índices de Cliente y Distribuidor.
- **`distribuidor`**: Elimina de `dataSet/` y actualiza índice de Distribuidor.

### Ejemplos

**Eliminar documento público:**
```bash
curl -X POST http://localhost:5000/api/delete \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "manual_obsoleto.md",
    "target": "cliente"
  }'
```

**Respuesta Exitosa:**
```json
{
  "success": true,
  "message": "File deleted successfully",
  "filename": "manual_obsoleto.md",
  "target": "cliente"
}
```

---

## ⚠️ Notas Importantes

1. **Extensiones**: El sistema rechazará cualquier archivo que no sea `.txt` o `.md`.
2. **Duplicados**: Si subes un archivo con el mismo contenido, se agregará nuevamente a la base de vectores (duplicando información). Es recomendable no resubir el mismo archivo sin cambios.
3. **Persistencia**: Los archivos subidos se guardan físicamente en el servidor en las carpetas `dataSet/` o `dataSet/datasetGen/` según el target.
