# Guía de Python

## Introducción a Python

Python es un lenguaje de programación de alto nivel, interpretado y de propósito general. Fue creado por Guido van Rossum y lanzado por primera vez en 1991.

### Características principales

- **Sintaxis clara y legible**: Python usa indentación para definir bloques de código
- **Tipado dinámico**: No es necesario declarar tipos de variables
- **Multiparadigma**: Soporta programación orientada a objetos, funcional e imperativa
- **Gran ecosistema**: Miles de librerías disponibles en PyPI

## Tipos de Datos Básicos

### Números

Python soporta varios tipos numéricos:

```python
entero = 42
flotante = 3.14
complejo = 1 + 2j
```

### Strings

Las cadenas de texto pueden definirse con comillas simples o dobles:

```python
nombre = "Python"
mensaje = 'Hola mundo'
multilinea = """
Este es un texto
de varias líneas
"""
```

### Listas

Las listas son colecciones ordenadas y mutables:

```python
frutas = ["manzana", "banana", "naranja"]
frutas.append("uva")
print(frutas[0])  # manzana
```

### Diccionarios

Los diccionarios almacenan pares clave-valor:

```python
persona = {
    "nombre": "Juan",
    "edad": 30,
    "ciudad": "Madrid"
}
print(persona["nombre"])  # Juan
```

## Estructuras de Control

### Condicionales

```python
edad = 18

if edad >= 18:
    print("Es mayor de edad")
elif edad >= 13:
    print("Es adolescente")
else:
    print("Es menor de edad")
```

### Bucles

**Bucle for:**

```python
for i in range(5):
    print(i)

for fruta in frutas:
    print(fruta)
```

**Bucle while:**

```python
contador = 0
while contador < 5:
    print(contador)
    contador += 1
```

## Funciones

Las funciones se definen con la palabra clave `def`:

```python
def saludar(nombre):
    """Función que saluda a una persona."""
    return f"Hola, {nombre}!"

mensaje = saludar("Ana")
print(mensaje)  # Hola, Ana!
```

### Funciones Lambda

```python
cuadrado = lambda x: x ** 2
print(cuadrado(5))  # 25
```

## Clases y Objetos

Python es un lenguaje orientado a objetos:

```python
class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad
    
    def presentarse(self):
        return f"Soy {self.nombre} y tengo {self.edad} años"

juan = Persona("Juan", 30)
print(juan.presentarse())
```

## Manejo de Excepciones

```python
try:
    resultado = 10 / 0
except ZeroDivisionError:
    print("No se puede dividir entre cero")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("Esto siempre se ejecuta")
```

## Módulos y Paquetes

Para importar módulos:

```python
import math
print(math.sqrt(16))  # 4.0

from datetime import datetime
print(datetime.now())
```

## Recursos Adicionales

- Documentación oficial: https://docs.python.org
- Tutorial de Python: https://docs.python.org/3/tutorial/
- PyPI (repositorio de paquetes): https://pypi.org
