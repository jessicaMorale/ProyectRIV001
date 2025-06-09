# Importa las librerías necesarias
import ir_datasets  # Para cargar conjuntos de datos de información retrieval
import re  # Para expresiones regulares
import nltk  # Natural Language Toolkit para procesamiento de lenguaje natural
from collections import Counter, defaultdict  # Contadores y diccionarios con valores por defecto
from sklearn.feature_extraction.text import TfidfVectorizer  # Para modelo TF-IDF
from sklearn.metrics.pairwise import cosine_similarity  # Para calcular similitud coseno
from nltk.corpus import stopwords  # Palabras vacías (stopwords)
from nltk.tokenize import word_tokenize  # Tokenizador de palabras
from rank_bm25 import BM25Okapi  # Implementación del algoritmo BM25
from nltk.stem import PorterStemmer  # Stemmer (reducción de palabras a su raíz)
from nltk.stem import WordNetLemmatizer  # Lematizador (mejor que stemming)
from nltk.corpus import wordnet  # Léxico WordNet para lematización
from tabulate import tabulate  # Para mostrar tablas formateadas
import pandas as pd

# Descargar recursos NLTK necesarios si no están disponibles
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        # Intenta encontrar los recursos
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        # Si no están disponibles, los descarga
        nltk.download(resource)
def show_queries_qrels(queries, qrels_dict, num_queries=20):
    """
    Muestra las queries y sus qrels correspondientes de forma organizada
    
    Args:
        queries: Lista de objetos query del dataset
        qrels_dict: Diccionario con qrels {query_id: [doc_ids]}
        num_queries: Número de queries a mostrar (por defecto 20)
    """
    print(f"\n🔍 QUERIES Y QRELS DEL DATASET")
    print("=" * 80)
    
    # Limitar el número de queries a mostrar
    queries_to_show = min(num_queries, len(queries))
    
    print(f"Total de queries: {len(queries)}")
    print(f"Queries con qrels: {len(qrels_dict)}")
    print(f"Mostrando las primeras {queries_to_show} queries:")
    print("-" * 80)
    
    # Contador para queries mostradas
    shown_queries = 0
    
    for i, query in enumerate(queries):
        if shown_queries >= queries_to_show:
            break
            
        print(f"\n📋 QUERY #{i+1}")
        print(f"   ID: {query.query_id}")
        print(f"   Texto: {query.text[:100]}{'...' if len(query.text) > 100 else ''}")
        
        # Mostrar qrels si existen para esta query
        if query.query_id in qrels_dict:
            relevant_docs = qrels_dict[query.query_id]
            print(f"   📄 Documentos relevantes ({len(relevant_docs)}):")
            
            # Mostrar hasta 10 documentos relevantes
            docs_to_show = min(10, len(relevant_docs))
            for j, doc_id in enumerate(relevant_docs[:docs_to_show]):
                print(f"      - {doc_id}")
            
            # Indicar si hay más documentos
            if len(relevant_docs) > docs_to_show:
                print(f"      ... y {len(relevant_docs) - docs_to_show} documentos más")
        else:
            print(f"   ❌ Sin qrels disponibles")
        
        print("   " + "-" * 60)
        shown_queries += 1
    
    # Mostrar estadísticas finales
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"   Total queries procesadas: {shown_queries}")
    print(f"   Queries con qrels: {sum(1 for q in queries[:queries_to_show] if q.query_id in qrels_dict)}")
    print(f"   Queries sin qrels: {sum(1 for q in queries[:queries_to_show] if q.query_id not in qrels_dict)}")
    
    # Mostrar distribución de qrels
    if qrels_dict:
        qrel_counts = [len(docs) for docs in qrels_dict.values()]
        print(f"   Promedio de docs relevantes por query: {sum(qrel_counts)/len(qrel_counts):.2f}")
        print(f"   Máximo de docs relevantes: {max(qrel_counts)}")
        print(f"   Mínimo de docs relevantes: {min(qrel_counts)}")
    
    input("\n📥 Presiona Enter para continuar...")
    
def get_wordnet_pos(word):
    """Mapea POS tags a WordNet POS tags para lemmatización"""
    tag = nltk.pos_tag([word])[0][1][0].upper()  # Obtiene la primera letra del POS tag
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}  # Mapeo de tags
    return tag_dict.get(tag, wordnet.NOUN)  # Devuelve el tag correspondiente o NOUN por defecto

def preprocess_text(text, use_stemming=False):
    """Función de preprocesamiento mejorada para texto técnico"""
    if not text:
        return ""  # Manejo de texto vacío
    
    # Preservar términos técnicos de programación reemplazándolos por versiones sin caracteres especiales
    programming_replacements = {
        r'\bC\+\+\b': 'cplusplus',
        r'\bC#\b': 'csharp', 
        r'\b\.NET\b': 'dotnet',
        r'\bNode\.js\b': 'nodejs',
        r'\bHTML5\b': 'html5',
        r'\bCSS3\b': 'css3'
    }
    
    # Aplica los reemplazos usando expresiones regulares
    for pattern, replacement in programming_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales (excepto +, -, # que son relevantes en programación)
    text = re.sub(r'[^\w\s\+\-\#]', ' ', text)
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenización (dividir el texto en palabras)
    tokens = word_tokenize(text)
    
    # Stopwords expandidas para programación
    programming_stopwords = {
        'code', 'example', 'question', 'answer', 'problem', 'need', 'want', 
        'know', 'help', 'please', 'thanks'
    }
    
    # Combinar stopwords tradicionales con las específicas de programación
    stop_words = set(stopwords.words('english')).union(programming_stopwords)
    # Filtrar stopwords y tokens muy cortos
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    # Usar lemmatización en lugar de stemming por defecto
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    else:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    
    # Unir los tokens nuevamente en un string
    return ' '.join(tokens)

def build_inverted_index(processed_docs):
    """Construye un índice invertido a partir de documentos preprocesados"""
    print("🔄 Construyendo índice invertido...")
    inverted_index = {}  # Diccionario para el índice invertido
    
    for doc_id, doc in enumerate(processed_docs):
        if not doc.strip():
            continue  # Saltar documentos vacíos
        
        # Contar frecuencia de términos en el documento
        term_freq = Counter(doc.split())
        
        # Agregar al índice invertido
        for term, freq in term_freq.items():
            if term not in inverted_index:
                inverted_index[term] = {}  # Inicializar entrada para el término
            inverted_index[term][doc_id] = freq  # Almacenar frecuencia en el documento
    
    print(f"✅ Índice invertido construido con {len(inverted_index)} términos únicos")
    return inverted_index

def build_tfidf_model(processed_docs):
    """Construye modelo TF-IDF usando sklearn"""
    print("🔄 Construyendo modelo TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer()  # Inicializar vectorizador
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)  # Ajustar y transformar documentos
    print("✅ Modelo TF-IDF construido")
    return tfidf_vectorizer, tfidf_matrix  # Devuelve vectorizador y matriz TF-IDF

def build_bm25_model(processed_docs):
    """Construye modelo BM25 usando rank_bm25"""
    print("🔄 Construyendo modelo BM25...")
    tokenized_docs = [doc.split() for doc in processed_docs]  # Tokenizar documentos para BM25
    bm25_model = BM25Okapi(tokenized_docs)  # Crear modelo BM25
    print("✅ Modelo BM25 construido")
    return bm25_model

def calculate_precision_recall(retrieved_docs, relevant_docs):
    """Calcula Precision y Recall para resultados de búsqueda"""
    retrieved_set, relevant_set = set(retrieved_docs), set(relevant_docs)  # Convertir a conjuntos
    relevant_retrieved = retrieved_set.intersection(relevant_set)  # Intersección de relevantes y recuperados
    
    # Calcular precision: fracción de recuperados que son relevantes
    precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0.0
    # Calcular recall: fracción de relevantes que fueron recuperados
    recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0.0
    
    return precision, recall

def calculate_average_precision(retrieved_docs, relevant_docs):
    """Calcula Average Precision para una query"""
    if not relevant_docs:
        return 0.0  # Si no hay documentos relevantes, AP es 0
    
    relevant_set = set(relevant_docs)  # Convertir a conjunto para búsqueda rápida
    precision_sum = 0.0  # Acumulador para sumar precisiones
    relevant_found = 0  # Contador de relevantes encontrados
    
    # Iterar sobre los documentos recuperados en orden
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_set:
            relevant_found += 1
            precision_at_i = relevant_found / i  # Precision en este punto
            precision_sum += precision_at_i  # Acumular
    
    # Devolver el promedio de las precisiones en los puntos donde se encontraron relevantes
    return precision_sum / len(relevant_docs) if relevant_docs else 0.0

def calculate_map_score(tfidf_vectorizer, tfidf_matrix, bm25_model, queries, qrels_dict, doc_id_to_index, processed_docs, method="tfidf", k=10):
    """Calcula MAP (Mean Average Precision) para todo el sistema"""
    print(f"🔄 Calculando MAP para {method.upper()}...")
    
    total_ap = 0.0  # Acumulador para Average Precisions
    evaluated_queries = 0  # Contador de queries evaluadas
    
    for query in queries:
        if query.query_id not in qrels_dict:
            continue  # Saltar queries sin relevancias definidas
        
        # Obtener índices de documentos relevantes para esta query
        relevant_indices = [doc_id_to_index[doc_id] for doc_id in qrels_dict[query.query_id] 
                          if doc_id in doc_id_to_index]
        
        if not relevant_indices:
            continue  # Saltar si no hay relevantes mapeados
        
        # Preprocesar el texto de la query
        processed_query = preprocess_text(query.text)
        if not processed_query.strip():
            continue  # Saltar queries vacías después de preprocesamiento
        
        # Obtener resultados según el método seleccionado
        if method == "tfidf":
            # Transformar query a vector TF-IDF
            query_vector = tfidf_vectorizer.transform([processed_query])
            # Calcular similitud coseno con todos los documentos
            cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
            # Filtrar y ordenar resultados
            results = [(i, score) for i, score in enumerate(cosine_sim) if score > 0]
            results.sort(key=lambda x: x[1], reverse=True)
            retrieved_indices = [doc_idx for doc_idx, score in results[:k]]
        else:  # BM25
            query_tokens = processed_query.split()
            # Obtener scores BM25 para la query
            scores = bm25_model.get_scores(query_tokens)
            # Filtrar y ordenar resultados
            results = [(i, score) for i, score in enumerate(scores) if score > 0]
            results.sort(key=lambda x: x[1], reverse=True)
            retrieved_indices = [doc_idx for doc_idx, score in results[:k]]
        
        if not retrieved_indices:
            continue  # Saltar si no se recuperaron documentos
        
        # Calcular Average Precision para esta query
        ap = calculate_average_precision(retrieved_indices, relevant_indices)
        total_ap += ap
        evaluated_queries += 1
        
        # Mostrar progreso cada 50 queries
        if evaluated_queries % 50 == 0:
            print(f"   Procesadas {evaluated_queries} queries...")
    
    # Calcular MAP como el promedio de los APs
    map_score = total_ap / evaluated_queries if evaluated_queries > 0 else 0.0
    return map_score, evaluated_queries

def show_results(query, results, docs, doc_ids, method_name):
    """Muestra resultados de búsqueda de forma formateada"""
    print(f"\n📋 RESULTADOS {method_name} PARA: '{query}'")
    print(f"Se encontraron {len(results)} documentos relevantes")
    print("=" * 60)
    
    # Mostrar los primeros 5 resultados
    for rank, (doc_id, score) in enumerate(results[:5], start=1):
        print(f"\n🔸 RESULTADO #{rank}")
        print(f"   Puntuación {method_name}: {score:.4f}")
        print(f"   ID Documento: {doc_ids[doc_id]}")
        # Mostrar un fragmento del contenido (primeros 200 caracteres)
        content = docs[doc_id][:200].replace('\n', ' ')
        print(f"   Contenido: {content}...")

        # Separador entre resultados (excepto el último)
        if rank < 5 and rank < len(results):
            print("   " + "─" * 50)
    
    # Indicar si hay más resultados
    if len(results) > 5:
        print(f"\n   📄 ... y {len(results) - 5} documentos más")

def search_tfidf(tfidf_vectorizer, tfidf_matrix, processed_query):
    """Realiza búsqueda usando TF-IDF"""
    # Transformar query a vector TF-IDF
    query_vector = tfidf_vectorizer.transform([processed_query])
    # Calcular similitud coseno con todos los documentos
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    # Filtrar resultados con score > 0 y ordenar por score descendente
    results = [(i, score) for i, score in enumerate(cosine_sim) if score > 0]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def search_bm25(bm25_model, processed_query):
    """Realiza búsqueda usando BM25"""
    # Tokenizar la query
    query_tokens = processed_query.split()
    # Obtener scores BM25 para todos los documentos
    scores = bm25_model.get_scores(query_tokens)
    # Filtrar resultados con score > 0 y ordenar por score descendente
    results = [(i, score) for i, score in enumerate(scores) if score > 0]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def show_documents_table(docs, processed_docs, num_docs=100):
    """
    Muestra una tabla comparativa de documentos originales vs preprocesados
    con formato mejorado para mejor legibilidad
    """
    # Limitar al número de documentos solicitado
    num_docs = min(num_docs, len(docs))
    
    # Preparar los datos para la tabla
    table_data = []
    for idx in range(num_docs):
        # Limitar y limpiar el texto original
        original_text = docs[idx][:100].replace('\n', ' ').replace('\t', ' ')
        if len(docs[idx]) > 100:
            original_text += "..."
        
        # Limitar y limpiar el texto procesado
        processed_text = processed_docs[idx][:100].replace('\n', ' ').replace('\t', ' ')
        if len(processed_docs[idx]) > 100:
            processed_text += "..."
        
        table_data.append([
            idx + 1,  # Índice/Número de documento
            original_text.strip(),  # Texto original limpio
            processed_text.strip()  # Texto procesado limpio
        ])
    
    # Configurar encabezados de la tabla
    headers = [
        "N° Doc", 
        "Texto Original (primeros 100 caracteres)", 
        "Texto Preprocesado (primeros 100 caracteres)"
    ]
    
    # Configuración de formato de tabla
    table_format = {
        "tablefmt": "grid",  # Usar bordes de tabla
        "maxcolwidths": [8, 45, 45],  # Ancho máximo de columnas
        "stralign": ["center", "left", "left"],  # Alineación del texto
        "numalign": "center"  # Alineación de números
    }
    
    # Mostrar la tabla con formato mejorado
    print("\n📋 TABLA COMPARATIVA: DOCUMENTOS ORIGINALES VS PREPROCESADOS")
    print("=" * 120)
    print(tabulate(table_data, headers=headers, **table_format))
    print(f"\nMostrando {num_docs} de {len(docs)} documentos disponibles")
    input("\n📥 Presiona Enter para continuar...")

def search_interface(tfidf_vectorizer, tfidf_matrix, bm25_model, docs, doc_ids, method_name):
    """Interfaz de búsqueda unificada para TF-IDF o BM25"""
    while True:
        print(f"\n🔍 BÚSQUEDA CON {method_name}")
        print("=" * 60)
        query = input("Escribe tu consulta (o 'atras' para volver): ").strip()
        
        if query.lower() == 'atras':
            break  # Volver al menú principal
        if not query:
            print("❌ Por favor ingresa una consulta válida.")
            continue

        # Preprocesar la consulta
        processed_query = preprocess_text(query)
        if not processed_query.strip():
            print("❌ La consulta no contiene términos válidos.")
            continue
        
        # Obtener resultados según el método seleccionado
        if method_name == "TF-IDF":
            results = search_tfidf(tfidf_vectorizer, tfidf_matrix, processed_query)
        else:  # BM25
            results = search_bm25(bm25_model, processed_query)
        
        if not results:
            print("❌ No se encontraron documentos relevantes.")
            continue
        
        # Mostrar resultados
        show_results(query, results, docs, doc_ids, method_name)
        input("\n📥 Presiona Enter para continuar...")

def evaluation_interface(tfidf_vectorizer, tfidf_matrix, bm25_model, queries, qrels_dict, doc_id_to_index, processed_docs):
    """Interfaz de evaluación del sistema"""
    print("\n📈 EVALUACIÓN DE RESULTADOS")
    print("=" * 60)
    
    method_option = input("Selecciona método (1-TF-IDF, 2-BM25, 3-MAP Completo, 4-Volver): ").strip()
    if method_option == "4" or method_option not in ["1", "2", "3"]:
        return  # Volver al menú principal
    
    if method_option == "3":
        # Calcular MAP para ambos métodos
        print("\n🎯 CALCULANDO MAP PARA TODO EL SISTEMA")
        print("=" * 60)
        
        # MAP para TF-IDF
        map_tfidf, queries_tfidf = calculate_map_score(tfidf_vectorizer, tfidf_matrix, bm25_model, queries, qrels_dict, doc_id_to_index, processed_docs, "tfidf")
        
        # MAP para BM25
        map_bm25, queries_bm25 = calculate_map_score(tfidf_vectorizer, tfidf_matrix, bm25_model, queries, qrels_dict, doc_id_to_index, processed_docs, "bm25")
        
        # Mostrar resultados comparativos
        print(f"\n📊 RESULTADOS MAP DEL SISTEMA COMPLETO")
        print("=" * 60)
        print(f"TF-IDF:")
        print(f"   MAP Score: {map_tfidf:.4f}")
        print(f"   Queries evaluadas: {queries_tfidf}")
        print(f"\nBM25:")
        print(f"   MAP Score: {map_bm25:.4f}")
        print(f"   Queries evaluadas: {queries_bm25}")
        print(f"\n🏆 Mejor método: {'TF-IDF' if map_tfidf > map_bm25 else 'BM25'}")
        
        input("\n📥 Presiona Enter para continuar...")
        return
    
    # Evaluación de precisión y recall para un subconjunto de queries
    num_queries = len(queries)
    print(f"\n🔄 Evaluando {num_queries} queries...")
    
    total_precision = total_recall = evaluated_queries = 0
    
    for query in queries[:num_queries]:
        if query.query_id not in qrels_dict:
            continue  # Saltar queries sin relevancias definidas
        
        # Obtener índices de documentos relevantes
        relevant_indices = [doc_id_to_index[doc_id] for doc_id in qrels_dict[query.query_id] 
                          if doc_id in doc_id_to_index]
        
        if not relevant_indices:
            continue  # Saltar si no hay relevantes mapeados
        
        # Preprocesar la query
        processed_query = preprocess_text(query.text)
        if not processed_query.strip():
            continue  # Saltar queries vacías después de preprocesamiento
        
        # Ejecutar búsqueda según el método seleccionado
        if method_option == "1":  # TF-IDF
            results = search_tfidf(tfidf_vectorizer, tfidf_matrix, processed_query)
            retrieved_indices = [doc_idx for doc_idx, score in results[:10]]  # Top 10
        else:  # BM25
            results = search_bm25(bm25_model, processed_query)
            retrieved_indices = [doc_idx for doc_idx, score in results[:10]]  # Top 10
        
        if not retrieved_indices:
            continue  # Saltar si no se recuperaron documentos
        
        # Calcular métricas
        precision, recall = calculate_precision_recall(retrieved_indices, relevant_indices)
        total_precision += precision
        total_recall += recall
        evaluated_queries += 1
        
        # Mostrar progreso cada 10 queries
        if evaluated_queries % 10 == 0:
            print(f"   Procesadas {evaluated_queries} queries...")
    
    # Mostrar resultados de la evaluación
    if evaluated_queries > 0:
        method_name = "TF-IDF" if method_option == "1" else "BM25"
        print(f"\n📊 RESULTADOS DE EVALUACIÓN - {method_name}")
        print("=" * 60)
        print(f"Queries evaluadas: {evaluated_queries}")
        print(f"Precision promedio: {total_precision/evaluated_queries:.4f}")
        print(f"Recall promedio: {total_recall/evaluated_queries:.4f}")
    else:
        print("\n❌ No se pudieron evaluar queries.")
    
    input("\n📥 Presiona Enter para continuar...")

def show_inverted_index(inverted_index, doc_ids, num_terms=50):
    """
    Muestra el índice invertido usando DataFrames de pandas para mejor visualización
    
    Args:
        inverted_index: Diccionario con el índice invertido
        doc_ids: Lista de IDs de documentos
        num_terms: Número de términos a mostrar (por defecto 50)
    """
    print(f"\n📚 ÍNDICE INVERTIDO")
    print("=" * 80)
    
    # Preparar datos para el DataFrame principal
    term_data = []
    detailed_postings = []
    
    for term, doc_freqs in inverted_index.items():
        total_freq = sum(doc_freqs.values())  # Frecuencia total del término
        num_docs = len(doc_freqs)  # Número de documentos que contienen el término
        
        # Agregar datos del término al resumen
        term_data.append({
            'Término': term,
            'Frecuencia_Total': total_freq,
            'Num_Documentos': num_docs,
            'Frecuencia_Promedio': round(total_freq / num_docs, 2)
        })
        
        # Agregar postings detallados para análisis posterior
        for doc_idx, freq in doc_freqs.items():
            doc_id = doc_ids[doc_idx] if doc_idx < len(doc_ids) else f"Doc_{doc_idx}"
            detailed_postings.append({
                'Término': term,
                'Doc_ID': doc_id,
                'Doc_Index': doc_idx,
                'Frecuencia': freq
            })
    
    # Crear DataFrame principal con resumen de términos
    df_terms = pd.DataFrame(term_data)
    df_terms = df_terms.sort_values('Frecuencia_Total', ascending=False)
    
    # Crear DataFrame con postings detallados
    df_postings = pd.DataFrame(detailed_postings)
    
    print(f"Total de términos únicos en el índice: {len(inverted_index)}")
    print(f"Total de postings: {len(df_postings)}")
    print("-" * 80)
    
    # Mostrar resumen de términos más frecuentes
    print(f"\n📊 RESUMEN DE LOS {min(num_terms, len(df_terms))} TÉRMINOS MÁS FRECUENTES:")
    print("-" * 80)
    
    # Configurar pandas para mostrar todas las columnas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)
    
    # Mostrar tabla resumen
    print(df_terms.head(num_terms).to_string(index=True, index_names=['Rank']))
    
    # Mostrar algunos ejemplos detallados de términos específicos
    print(f"\n🔍 EJEMPLOS DETALLADOS DE POSTINGS:")
    print("-" * 80)
    
    # Seleccionar algunos términos interesantes para mostrar en detalle
    sample_terms = df_terms.head(5)['Término'].tolist()
    
    for i, term in enumerate(sample_terms, 1):
        print(f"\n{i}. TÉRMINO: '{term}'")
        term_postings = df_postings[df_postings['Término'] == term].sort_values('Frecuencia', ascending=False)
        
        # Mostrar estadísticas del término
        stats = df_terms[df_terms['Término'] == term].iloc[0]
        print(f"   📊 Frecuencia total: {stats['Frecuencia_Total']}")
        print(f"   📄 Aparece en {stats['Num_Documentos']} documentos")
        print(f"   📈 Frecuencia promedio: {stats['Frecuencia_Promedio']}")
        
        # Mostrar postings (máximo 10)
        print(f"   🔍 Top documentos:")
        display_postings = term_postings.head(10)
        for _, posting in display_postings.iterrows():
            print(f"      - {posting['Doc_ID']}: {posting['Frecuencia']} veces")
        
        if len(term_postings) > 10:
            print(f"      ... y {len(term_postings) - 10} documentos más")
        
        print("   " + "-" * 60)
    
    # Mostrar estadísticas generales del índice
    print(f"\n📈 ESTADÍSTICAS GENERALES DEL ÍNDICE:")
    print("-" * 80)
    
    # Crear DataFrame con estadísticas
    stats_data = {
        'Métrica': [
            'Total de términos únicos',
            'Total de postings',
            'Promedio de términos por documento',
            'Promedio de documentos por término',
            'Término más frecuente',
            'Frecuencia máxima',
            'Término en más documentos',
            'Máximo de documentos por término'
        ],
        'Valor': [
            len(df_terms),
            len(df_postings),
            round(len(df_postings) / len(doc_ids), 2),
            round(df_terms['Num_Documentos'].mean(), 2),
            df_terms.iloc[0]['Término'],
            df_terms.iloc[0]['Frecuencia_Total'],
            df_terms.loc[df_terms['Num_Documentos'].idxmax(), 'Término'],
            df_terms['Num_Documentos'].max()
        ]
    }
    
    df_stats = pd.DataFrame(stats_data)
    print(df_stats.to_string(index=False))
    

    
    # Restablecer opciones de pandas
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    input("\n📥 Presiona Enter para continuar...")

def show_dataset_stats(docs, queries, qrels_dict, inverted_index):
    """Muestra estadísticas del dataset"""
    print(f"\n📊 INFORMACION DEL DATASET:")
    print("=" * 60)
    print(f"   Total de documentos: {len(docs)}")
    print(f"   Total de queries: {len(queries)}")
    print(f"   Queries con qrels: {len(qrels_dict)}")
    print(f"   Términos únicos: {len(inverted_index)}")
    print(f"   Total de postings: {sum(len(docs) for docs in inverted_index.values())}")
    
    input("\n📥 Presiona Enter para continuar...")

def main():
    """Función principal que coordina todo el sistema"""
    # Carga del dataset
    print("🔄 Cargando dataset BEIR CQADupStack programmers...")
    dataset = ir_datasets.load("beir/cqadupstack/programmers")  # Cargar dataset específico
    
    # Extraer datos del dataset
    docs, doc_ids, doc_id_to_index = [], [], {}  # Listas para almacenar documentos y mapeos
    
    # Procesar cada documento del dataset
    for idx, doc in enumerate(dataset.docs_iter()):
        # Combinar título y texto (si existen)
        text = (doc.title + " " if hasattr(doc, 'title') and doc.title else "") + \
               (doc.text if hasattr(doc, 'text') and doc.text else "")
        docs.append(text.strip())  # Almacenar texto del documento
        doc_ids.append(doc.doc_id)  # Almacenar ID del documento
        doc_id_to_index[doc.doc_id] = idx  # Mapear ID a índice
    
    # Obtener todas las queries del dataset
    queries = list(dataset.queries_iter())
    
    # Procesar qrels (relevancias) para evaluaciones
    qrels_dict = defaultdict(list)  # Diccionario para almacenar qrels
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:  # Solo considerar relevancias positivas
            qrels_dict[qrel.query_id].append(qrel.doc_id)  # Agregar a qrels
    
    print(f"✅ Dataset cargado: {len(docs)} documentos, {len(queries)} queries, {len(qrels_dict)} qrels.")
    
    # Preprocesamiento y construcción de modelos
    print("🔄 Procesando documentos...")
    processed_docs = [preprocess_text(doc) for doc in docs]  # Preprocesar todos los documentos
    inverted_index = build_inverted_index(processed_docs)  # Construir índice invertido
    
    # Construir modelos de recuperación
    tfidf_vectorizer, tfidf_matrix = build_tfidf_model(processed_docs)  # Modelo TF-IDF
    bm25_model = build_bm25_model(processed_docs)  # Modelo BM25
    
    print("🎉 Sistema listo!")
    
    # Menú principal interactivo
    while True:
        print("\n" + "="*60)
        print("🔍 SISTEMA DE RECUPERACION DE INFORMACION")
        print("="*60)
        # Opciones del menú
        options = [
            "📋 Ver Informacion del Dataset",
            "📚 Ver Índice Invertido",
            "📄 Mostrar tabla de documentos (original vs procesado)",
            "🔍 Búsqueda TF-IDF", 
            "🎯 Búsqueda BM25", 
            "📈 Evaluación de Resultados", 
            "❌ Salir",
            "🔍 Ver Queries y QRels"
        ]
        # Mostrar opciones numeradas
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        print("="*60)
        
        # Obtener selección del usuario
        option = input("\nSelecciona una opción (1-7): ").strip()
        
        # Manejar la opción seleccionada
        if option == "1":
            show_dataset_stats(docs, queries, qrels_dict, inverted_index)
        elif option == "2":
            show_inverted_index(inverted_index, doc_ids)  # ← AGREGAR ESTA LÍNEA
        elif option == "3":  # Los números cambian6
            show_documents_table(docs, processed_docs)
        elif option == "4":
            search_interface(tfidf_vectorizer, tfidf_matrix, bm25_model, docs, doc_ids, "TF-IDF")
        elif option == "5":
            search_interface(tfidf_vectorizer, tfidf_matrix, bm25_model, docs, doc_ids, "BM25")
        elif option == "6":
            evaluation_interface(tfidf_vectorizer, tfidf_matrix, bm25_model, queries, qrels_dict, doc_id_to_index, processed_docs)
        elif option == "8":  # Nueva opción
            show_queries_qrels(queries, qrels_dict)
        elif option == "7":  # Cambiar a 7
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("\n❌ Opción no válida. Por favor selecciona 1-7.") 

if __name__ == "__main__":
    main()  # Ejecutar la función principal si el script se ejecuta directamente
