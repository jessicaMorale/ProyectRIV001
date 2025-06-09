# Sistema de Recuperación de Información

Un sistema completo de recuperación de información que implementa algoritmos TF-IDF y BM25 para búsqueda y evaluación de documentos técnicos de programación.

## 🎯 Características Principales

- **Múltiples Algoritmos**: Implementación de TF-IDF y BM25
- **Preprocesamiento Avanzado**: Optimizado para contenido técnico de programación
- **Evaluación Completa**: Métricas MAP, Precision y Recall
- **Interfaz Interactiva**: Menú de consola fácil de usar
- **Análisis Detallado**: Visualización de índices invertidos y estadísticas

## 📋 Requisitos del Sistema

### Dependencias de Python

```bash
pip install ir-datasets nltk scikit-learn rank-bm25 tabulate pandas
```

### Recursos NLTK

El sistema descarga automáticamente los recursos necesarios:
- `punkt` y `punkt_tab` (tokenización)
- `stopwords` (palabras vacías)
- `wordnet` (lematización)

## 🚀 Instalación

1. **Clonar o descargar el código**
   ```bash
   # Si tienes el archivo
   python sistema_recuperacion.py
   ```

2. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el sistema**
   ```bash
   python sistema_recuperacion.py
   ```

## 🔧 Uso del Sistema

### Menú Principal

Al ejecutar el sistema, verás las siguientes opciones:

```
🔍 SISTEMA DE RECUPERACION DE INFORMACION
============================================================
1. 📋 Ver Información del Dataset
2. 📚 Ver Índice Invertido
3. 📄 Mostrar tabla de documentos (original vs procesado)
4. 🔍 Búsqueda TF-IDF
5. 🎯 Búsqueda BM25
6. 📈 Evaluación de Resultados
7. ❌ Salir
8. 🔍 Ver Queries y QRels
```

### Funcionalidades Detalladas

#### 1. Información del Dataset
- Estadísticas generales del corpus
- Número de documentos, queries y relevancias
- Información del índice invertido

#### 2. Índice Invertido
- Visualización de términos y sus frecuencias
- Postings detallados por término
- Estadísticas del vocabulario

#### 3. Comparación de Documentos
- Vista lado a lado: texto original vs preprocesado
- Formato tabular para fácil comparación
- Muestra los primeros 100 caracteres de cada documento

#### 4. Búsqueda TF-IDF
- Interfaz interactiva de búsqueda
- Resultados ordenados por similitud coseno
- Muestra los top 5 documentos más relevantes

#### 5. Búsqueda BM25
- Implementación del algoritmo BM25 Okapi
- Búsqueda probabilística
- Resultados con puntuaciones BM25

#### 6. Evaluación de Resultados
- Cálculo de MAP (Mean Average Precision)
- Métricas de Precision y Recall
- Comparación entre TF-IDF y BM25

#### 7. Queries y QRels
- Visualización de consultas del dataset
- Documentos relevantes asociados
- Estadísticas de relevancias

## 🛠️ Procesamiento de Texto

### Características del Preprocesamiento

- **Preservación de términos técnicos**: C++, C#, .NET, Node.js, etc.
- **Normalización**: Conversión a minúsculas y limpieza de caracteres
- **Tokenización**: División inteligente del texto
- **Filtrado de stopwords**: Incluye stopwords específicas de programación
- **Lematización**: Reducción de palabras a su forma canónica

### Ejemplo de Transformación

```
Original: "How to use C++ for Node.js development?"
Procesado: "cplusplus nodejs development"
```

## 📊 Métricas de Evaluación

### Mean Average Precision (MAP)
Promedio de las precisiones medias de todas las queries evaluadas.

### Precision y Recall
- **Precision**: Fracción de documentos recuperados que son relevantes
- **Recall**: Fracción de documentos relevantes que fueron recuperados

### Fórmulas

```
Precision = |Relevantes ∩ Recuperados| / |Recuperados|
Recall = |Relevantes ∩ Recuperados| / |Relevantes|
AP = Σ(Precision@k × rel(k)) / |Relevantes|
MAP = Σ(AP) / |Queries|
```

## 🗂️ Dataset

El sistema utiliza el dataset **BEIR CQADupStack Programmers**:
- **Dominio**: Preguntas y respuestas de programación
- **Tamaño**: Variable según la versión
- **Formato**: Documentos con título y contenido
- **Queries**: Consultas reales de usuarios
- **QRels**: Relevancias ground truth

## 🔍 Algoritmos Implementados

### TF-IDF (Term Frequency-Inverse Document Frequency)
- Vectorización usando scikit-learn
- Similitud coseno para ranking
- Eficiente para colecciones medianas

### BM25 (Best Matching 25)
- Implementación Okapi BM25
- Función de ranking probabilística
- Parámetros k1=1.2, b=0.75 (valores estándar)

## 📁 Estructura del Código

```
sistema_recuperacion.py
├── main()                          # Función principal
├── Carga de datos
│   ├── dataset.docs_iter()        # Iterador de documentos
│   ├── dataset.queries_iter()     # Iterador de queries
│   └── dataset.qrels_iter()       # Iterador de relevancias
├── Preprocesamiento
│   ├── preprocess_text()          # Función principal de preprocesamiento
│   └── get_wordnet_pos()          # Mapeo POS para lematización
├── Construcción de modelos
│   ├── build_inverted_index()     # Índice invertido
│   ├── build_tfidf_model()        # Modelo TF-IDF
│   └── build_bm25_model()         # Modelo BM25
├── Búsqueda
│   ├── search_tfidf()             # Búsqueda TF-IDF
│   └── search_bm25()              # Búsqueda BM25
├── Evaluación
│   ├── calculate_precision_recall()
│   ├── calculate_average_precision()
│   └── calculate_map_score()
└── Interfaces de usuario
    ├── search_interface()         # Interfaz de búsqueda
    ├── evaluation_interface()     # Interfaz de evaluación
    └── show_*()                   # Funciones de visualización
```

## ⚡ Rendimiento

### Optimizaciones
- Índice invertido en memoria para acceso rápido
- Vectorización eficiente con scipy.sparse
- Filtrado de resultados con score > 0

## 🎨 Personalización

### Modificar Preprocesamiento
```python
def preprocess_text(text, use_stemming=False):
    # Agregar términos técnicos específicos
    programming_replacements = {
        r'\bPython3\b': 'python3',
        r'\bReact\.js\b': 'reactjs',
        # Agregar más...
    }
```

### Ajustar Parámetros BM25
```python
# En build_bm25_model()
bm25_model = BM25Okapi(tokenized_docs, k1=1.2, b=0.75)
```

### Cambiar Dataset
```python
# En main()
dataset = ir_datasets.load("beir/otro-dataset")
```

## 🐛 Solución de Problemas

### Errores Comunes

1. **ModuleNotFoundError**: Instalar dependencias faltantes
   ```bash
   pip install [nombre-del-paquete]
   ```

2. **NLTK Data Error**: Los recursos se descargan automáticamente
   ```python
   nltk.download('punkt')
   ```

3. **Memory Error**: Reducir tamaño del dataset o aumentar RAM

4. **Encoding Issues**: Asegurar que los archivos estén en UTF-8

### Logs y Debug
El sistema muestra progreso detallado:
- ✅ Operaciones completadas exitosamente
- 🔄 Operaciones en progreso
- ❌ Errores o advertencias

## 📈 Interpretación de Resultados

### Puntuaciones TF-IDF
- Rango: 0.0 - 1.0
- Valores más altos = mayor similitud
- Típicamente: > 0.1 es relevante

### Puntuaciones BM25
- Rango: 0.0 - ∞
- Valores más altos = mayor relevancia
- Típicamente: > 2.0 es relevante

### MAP Scores
- Rango: 0.0 - 1.0
- > 0.3: Sistema decente
- > 0.5: Sistema bueno
- > 0.7: Sistema excelente

## 🤝 Contribuciones

Para mejorar el sistema:

1. **Optimizaciones de rendimiento**
2. **Nuevos algoritmos de ranking**
3. **Mejores técnicas de preprocesamiento**
4. **Interfaces gráficas**
5. **Soporte para más datasets**

