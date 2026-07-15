# Visualización de Espacios Semánticos y NLP: El Viejo y El Mar

Este proyecto implementa un pipeline modular de Procesamiento de Lenguaje Natural (NLP) para procesar texto, entrenar modelos de semántica distribucional y visualizar espacios vectoriales en 3D.

A diferencia de las representaciones clásicas (como Bag of Words o TF-IDF) que sufren de alta dimensionalidad y pérdida de contexto, este sistema utiliza Word Embeddings (Word2Vec) para capturar relaciones semánticas complejas y las proyecta en un espacio tridimensional legible mediante PCA (Análisis de Componentes Principales), aislando vecindarios semánticos específicos para evitar el overplotting y la colisión de etiquetas.

## 🛠️ Arquitectura del Proyecto

El código está estructurado bajo el principio de Responsabilidad Única, separando estrictamente la lógica matemática y de extracción de los componentes de renderizado gráfico.

```Plaintext
mi_proyecto/
│
├── main.ipynb                 # Orquestador principal (Notebook)
├── libro.txt                  # Dataset de texto (El Viejo y El Mar)
└── utils/                     # Paquete de herramientas modulares
    ├── __init__.py            # Inicializador de paquete Python
    ├── feature_extraction.py  # Lógica matemática, NLP y Embeddings
    └── visualization_3d.py    # Lógica exclusiva de gráficos en Matplotlib
```
# 🚀 Componentes del Sistema
## 1. Extracción de Características e IA (utils/feature_extraction.py)

Contiene el pipeline de limpieza profunda de texto y el entrenamiento del modelo neuronal Word2Vec con soporte multihilo.

### 1. Pipeline NLP (spaCy):

- Carga Segura: Verifica la existencia de es_core_news_sm y controla su inicialización.

- Filtrado Semántico: Discrimina y elimina signos de puntuación y stop words.

- Lematización: Convierte verbos a su infinitivo y sustantivos/adjetivos a su forma raíz en minúsculas (ej. pescaba ➡ pescar).

### 2. Funciones del Módulo:

- `procesar_corpus_oraciones(texto)`: Normaliza, lematiza y limpia el texto crudo.

- `entrenar_modelo_word2vec(oraciones, dimensiones)`: Entrena la red neuronal y genera los vectores densos.

- `aplicar_pca_embeddings(modelo, n_componentes)`: Reduce las dimensiones latentes del modelo a componentes 3D.

- `obtener_vecindario_pca(modelo, df_pca, palabra_objetivo, top_n)`: Filtra el espacio global para extraer únicamente los 15 vecinos más cercanos a un concepto.

# 2. Capa de Visualización (utils/visualization_3d.py)

## Módulo encargado exclusivamente del renderizado en lienzos tridimensionales de Matplotlib.

- `graficar_vecindario_3d(ax, df_3d, palabra_objetivo, titulo)`: Genera el gráfico de dispersión 3D, mapeando la palabra objetivo en color rojo destacado (mayor escala) y sus vecinos semánticos en azul sutil con etiquetas de texto legibles.

### 💻 Ejemplo de Uso (Orquestación en main.ipynb)

El flujo principal se ejecuta de manera limpia y declarativa en el cuaderno de la siguiente forma:
```Python

import matplotlib.pyplot as plt
from utils.feature_extraction import (procesar_corpus_oraciones, 
                                      entrenar_modelo_word2vec, 
                                      aplicar_pca_embeddings,
                                      obtener_vecindario_pca)
from utils.visualization_3d import graficar_vecindario_3d

# 1. Cargar texto y procesar pipeline NLP
with open("libro.txt", "r", encoding="utf-8") as f:
    texto = f.read()

oraciones = procesar_corpus_oraciones(texto)

# 2. Entrenar modelo e interactuar con el espacio semántico
modelo_w2v = entrenar_modelo_word2vec(oraciones, dimensiones=10)
df_pca_3d = aplicar_pca_embeddings(modelo_w2v, n_componentes=3)

# 3. Aislar vecindario semántico de un concepto (Ej: "poder")
palabra_target = "poder"
df_vecindario = obtener_vecindario_pca(modelo_w2v, df_pca_3d, palabra_target, top_n=15)

# 4. Renderizar gráfico 3D limpio sin overplotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

graficar_vecindario_3d(ax, df_vecindario, palabra_target, f'Vecindario Semántico de "{palabra_target}"')
plt.show()
```
## 📦 Requisitos e Instalación
Tecnologías Utilizadas

    Python 3.12+

    spaCy (Procesamiento de Lenguaje Natural)

    Gensim (Modelado de Word Embeddings)

    Scikit-Learn (Reducción de dimensionalidad PCA)

    Matplotlib & Pandas (Visualización y estructuras de datos)

Instalación del Entorno

1. Clona este repositorio y asegúrate de tener tu entorno virtual activo.

2. Instala las dependencias necesarias:

```Bash

pip install -r requirements.txt
```
3. Descarga el modelo optimizado en español para spaCy:

```Bash

python -m spacy download es_core_news_sm
```