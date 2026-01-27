# FastAPI - Framework Web Moderno

## ¿Qué es FastAPI?

FastAPI es un framework web moderno y de alto rendimiento para construir APIs con Python 3.7+ basado en type hints estándar de Python.

### Características principales

- **Rápido**: Muy alto rendimiento, a la par con NodeJS y Go
- **Rápido de codificar**: Incrementa la velocidad de desarrollo entre 200% a 300%
- **Menos bugs**: Reduce cerca del 40% de errores inducidos por desarrolladores
- **Intuitivo**: Gran soporte de editor, autocompletado en todas partes
- **Fácil**: Diseñado para ser fácil de usar y aprender
- **Robusto**: Código listo para producción con documentación automática

## Instalación

```bash
pip install fastapi
pip install uvicorn[standard]
```

## Ejemplo Básico

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

Para ejecutar:

```bash
uvicorn main:app --reload
```

## Path Parameters

Los parámetros de ruta se definen directamente en el decorador:

```python
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}
```

## Query Parameters

Los parámetros de consulta se pasan automáticamente:

```python
@app.get("/items/")
def list_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

## Request Body con Pydantic

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

@app.post("/items/")
def create_item(item: Item):
    return item
```

## Validación Automática

FastAPI valida automáticamente los datos:

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    age: int = Field(..., ge=0, le=120)
```

## Documentación Automática

FastAPI genera documentación automática:

- **Swagger UI**: Disponible en `/docs`
- **ReDoc**: Disponible en `/redoc`

## Dependencias

El sistema de inyección de dependencias es muy potente:

```python
from fastapi import Depends

def get_db():
    db = DatabaseSession()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/")
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()
```

## Middleware

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Manejo de Errores

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]
```

## Async/Await

FastAPI soporta código asíncrono:

```python
@app.get("/async-items/")
async def read_items():
    results = await some_async_operation()
    return results
```

## Recursos

- Documentación oficial: https://fastapi.tiangolo.com
- GitHub: https://github.com/tiangolo/fastapi
