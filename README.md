# Procesamiento de Lenguaje Natural: Normalización y Lematización de Textos

Este proyecto implementa un pipeline fundamental de Procesamiento de Lenguaje Natural (NLP) utilizando **spaCy** en Python. Toma como entrada un archivo de texto crudo, lo limpia eliminando "ruido" gramatical y reduce las palabras a su raíz léxica (lematización) para facilitar análisis posteriores.

## 🚀 Características del proceso

1. **Carga Segura de Modelo:** Implementa un bloque de control que verifica la existencia del modelo en español (`es_core_news_sm`) y lo descarga automáticamente si el usuario no lo tiene instalado.
2. **Tokenización:** Divide el texto completo en unidades procesables (tokens) conservando los metadatos lingüísticos.
3. **Filtrado de Stop Words:** Discrimina y elimina signos de puntuación y palabras vacías (artículos, preposiciones) que no aportan peso semántico significativo.
4. **Lematización y Normalización:** Convierte verbos conjugados a su infinitivo y estandariza todo el texto a minúsculas (ej. *pescaba* ➡ *pescar*).

## 🛠️ Librerias Utilizadas

- **Python 3.12.3**
- **spaCy** (Librería principal de NLP)
- **es_core_news_sm** (Modelo de lenguaje optimizado para español)
- **Jupyter Notebook / IPython** (Entorno de desarrollo y experimentación)
