# Importa las librerías necesarias
import ir_datasets
import re, nltk
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

# Descargar recursos NLTK
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

def preprocess_text(text):
    """Procesamiento básico del texto: tokenización, normalización, remoción de stopwords"""
    if not text:
        return ""
    
    # Normalización y tokenización
    text = re.sub(r'[^a-z\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    
    # Remoción de stopwords
    stop_words = set(stopwords.words('english'))
    return ' '.join([token for token in tokens if token not in stop_words])

def build_inverted_index(processed_docs):
    """Construye un índice invertido"""
    print("🔄 Construyendo índice invertido...")
    inverted_index = {}
    
    for doc_id, doc in enumerate(processed_docs):
        if not doc.strip():
            continue
        
        # Contar frecuencia de términos
        term_freq = Counter(doc.split())
        
        # Agregar al índice invertido
        for term, freq in term_freq.items():
            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][doc_id] = freq
    
    print(f"✅ Índice invertido construido con {len(inverted_index)} términos únicos")
    return inverted_index

def build_tfidf_model(processed_docs):
    """Construye modelo TF-IDF usando sklearn"""
    print("🔄 Construyendo modelo TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)
    print("✅ Modelo TF-IDF construido")
    return tfidf_vectorizer, tfidf_matrix

def build_bm25_model(processed_docs):
    """Construye modelo BM25 usando rank_bm25"""
    print("🔄 Construyendo modelo BM25...")
    tokenized_docs = [doc.split() for doc in processed_docs]
    bm25_model = BM25Okapi(tokenized_docs)
    print("✅ Modelo BM25 construido")
    return bm25_model

def calculate_precision_recall(retrieved_docs, relevant_docs):
    """Calcula Precision y Recall"""
    retrieved_set, relevant_set = set(retrieved_docs), set(relevant_docs)
    relevant_retrieved = retrieved_set.intersection(relevant_set)
    
    precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0.0
    
    return precision, recall

def calculate_average_precision(retrieved_docs, relevant_docs):
    """Calcula Average Precision para una query"""
    if not relevant_docs:
        return 0.0
    
    relevant_set = set(relevant_docs)
    precision_sum = 0.0
    relevant_found = 0
    
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_set:
            relevant_found += 1
            precision_at_i = relevant_found / i
            precision_sum += precision_at_i
    
    return precision_sum / len(relevant_docs) if relevant_docs else 0.0

def calculate_map_score(tfidf_vectorizer, tfidf_matrix, bm25_model, queries, qrels_dict, doc_id_to_index, processed_docs, method="tfidf", k=10):
    """Calcula MAP (Mean Average Precision) para todo el sistema"""
    print(f"🔄 Calculando MAP para {method.upper()}...")
    
    total_ap = 0.0
    evaluated_queries = 0
    
    for query in queries:
        if query.query_id not in qrels_dict:
            continue
        
        relevant_indices = [doc_id_to_index[doc_id] for doc_id in qrels_dict[query.query_id] 
                          if doc_id in doc_id_to_index]
        
        if not relevant_indices:
            continue
        
        processed_query = preprocess_text(query.text)
        if not processed_query.strip():
            continue
        
        # Obtener resultados según el método
        if method == "tfidf":
            query_vector = tfidf_vectorizer.transform([processed_query])
            cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
            results = [(i, score) for i, score in enumerate(cosine_sim) if score > 0]
            results.sort(key=lambda x: x[1], reverse=True)
            retrieved_indices = [doc_idx for doc_idx, score in results[:k]]
        else:  # BM25
            query_tokens = processed_query.split()
            scores = bm25_model.get_scores(query_tokens)
            results = [(i, score) for i, score in enumerate(scores) if score > 0]
            results.sort(key=lambda x: x[1], reverse=True)
            retrieved_indices = [doc_idx for doc_idx, score in results[:k]]
        
        if not retrieved_indices:
            continue
        
        # Calcular Average Precision para esta query
        ap = calculate_average_precision(retrieved_indices, relevant_indices)
        total_ap += ap
        evaluated_queries += 1
        
        if evaluated_queries % 50 == 0:
            print(f"   Procesadas {evaluated_queries} queries...")
    
    map_score = total_ap / evaluated_queries if evaluated_queries > 0 else 0.0
    return map_score, evaluated_queries

def show_results(query, results, docs, doc_ids, method_name):
    """Muestra resultados de búsqueda"""
    print(f"\n📋 RESULTADOS {method_name} PARA: '{query}'")
    print(f"Se encontraron {len(results)} documentos relevantes")
    print("=" * 60)
    
    for rank, (doc_id, score) in enumerate(results[:5], start=1):
        print(f"\n🔸 RESULTADO #{rank}")
        print(f"   Puntuación {method_name}: {score:.4f}")
        print(f"   ID Documento: {doc_ids[doc_id]}")
        content = docs[doc_id][:200].replace('\n', ' ')
        print(f"   Contenido: {content}...")

        if rank < 5 and rank < len(results):
            print("   " + "─" * 50)
    
    if len(results) > 5:
        print(f"\n   📄 ... y {len(results) - 5} documentos más")

def search_tfidf(tfidf_vectorizer, tfidf_matrix, processed_query):
    """Búsqueda usando TF-IDF"""
    query_vector = tfidf_vectorizer.transform([processed_query])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    results = [(i, score) for i, score in enumerate(cosine_sim) if score > 0]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def search_bm25(bm25_model, processed_query):
    """Búsqueda usando BM25"""
    query_tokens = processed_query.split()
    scores = bm25_model.get_scores(query_tokens)
    results = [(i, score) for i, score in enumerate(scores) if score > 0]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def search_interface(tfidf_vectorizer, tfidf_matrix, bm25_model, docs, doc_ids, method_name):
    """Interfaz de búsqueda unificada"""
    while True:
        print(f"\n🔍 BÚSQUEDA CON {method_name}")
        print("=" * 60)
        query = input("Escribe tu consulta (o 'atras' para volver): ").strip()
        
        if query.lower() == 'atras':
            break
        if not query:
            print("❌ Por favor ingresa una consulta válida.")
            continue

        processed_query = preprocess_text(query)
        if not processed_query.strip():
            print("❌ La consulta no contiene términos válidos.")
            continue
        
        # Obtener resultados según el método
        if method_name == "TF-IDF":
            results = search_tfidf(tfidf_vectorizer, tfidf_matrix, processed_query)
        else:  # BM25
            results = search_bm25(bm25_model, processed_query)
        
        if not results:
            print("❌ No se encontraron documentos relevantes.")
            continue
        
        show_results(query, results, docs, doc_ids, method_name)
        input("\n📥 Presiona Enter para continuar...")

def evaluation_interface(tfidf_vectorizer, tfidf_matrix, bm25_model, queries, qrels_dict, doc_id_to_index, processed_docs):
    """Interfaz de evaluación"""
    print("\n📈 EVALUACIÓN DE RESULTADOS")
    print("=" * 60)
    
    method_option = input("Selecciona método (1-TF-IDF, 2-BM25, 3-MAP Completo, 4-Volver): ").strip()
    if method_option == "4" or method_option not in ["1", "2", "3"]:
        return
    
    if method_option == "3":
        # Calcular MAP para ambos métodos
        print("\n🎯 CALCULANDO MAP PARA TODO EL SISTEMA")
        print("=" * 60)
        
        # MAP para TF-IDF
        map_tfidf, queries_tfidf = calculate_map_score(tfidf_vectorizer, tfidf_matrix, bm25_model, queries, qrels_dict, doc_id_to_index, processed_docs, "tfidf")
        
        # MAP para BM25
        map_bm25, queries_bm25 = calculate_map_score(tfidf_vectorizer, tfidf_matrix, bm25_model, queries, qrels_dict, doc_id_to_index, processed_docs, "bm25")
        
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
    
    num_queries = len(queries)
    print(f"\n🔄 Evaluando {num_queries} queries...")
    
    total_precision = total_recall = evaluated_queries = 0
    
    for query in queries[:num_queries]:
        if query.query_id not in qrels_dict:
            continue
        
        relevant_indices = [doc_id_to_index[doc_id] for doc_id in qrels_dict[query.query_id] 
                          if doc_id in doc_id_to_index]
        
        if not relevant_indices:
            continue
        
        processed_query = preprocess_text(query.text)
        if not processed_query.strip():
            continue
        
        # Ejecutar búsqueda
        if method_option == "1":  # TF-IDF
            results = search_tfidf(tfidf_vectorizer, tfidf_matrix, processed_query)
            retrieved_indices = [doc_idx for doc_idx, score in results[:10]]
        else:  # BM25
            results = search_bm25(bm25_model, processed_query)
            retrieved_indices = [doc_idx for doc_idx, score in results[:10]]
        
        if not retrieved_indices:
            continue
        
        precision, recall = calculate_precision_recall(retrieved_indices, relevant_indices)
        total_precision += precision
        total_recall += recall
        evaluated_queries += 1
        
        if evaluated_queries % 10 == 0:
            print(f"   Procesadas {evaluated_queries} queries...")
    
    # Mostrar resultados
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

def show_dataset_stats(docs, queries, qrels_dict, inverted_index):
    """Muestra informacion del dataset"""
    print(f"\n📊 INFORMACION DEL DATASET:")
    print("=" * 60)
    print(f"   Total de documentos: {len(docs)}")
    print(f"   Total de queries: {len(queries)}")
    print(f"   Queries con qrels: {len(qrels_dict)}")
    print(f"   Términos únicos: {len(inverted_index)}")
    print(f"   Total de postings: {sum(len(docs) for docs in inverted_index.values())}")
    
    input("\n📥 Presiona Enter para continuar...")

def main():
    """Función principal"""
    # Carga del dataset
    print("🔄 Cargando dataset BEIR CQADupStack programmers...")
    dataset = ir_datasets.load("beir/cqadupstack/programmers")
    
    # Extraer datos
    docs, doc_ids, doc_id_to_index = [], [], {}
    
    for idx, doc in enumerate(dataset.docs_iter()):
        text = (doc.title + " " if hasattr(doc, 'title') and doc.title else "") + \
               (doc.text if hasattr(doc, 'text') and doc.text else "")
        docs.append(text.strip())
        doc_ids.append(doc.doc_id)
        doc_id_to_index[doc.doc_id] = idx
    
    queries = list(dataset.queries_iter())
    
    qrels_dict = defaultdict(list)
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels_dict[qrel.query_id].append(qrel.doc_id)
    
    print(f"✅ Dataset cargado: {len(docs)} documentos, {len(queries)} queries, {len(qrels_dict)} qrels.")
    
    # Preprocesamiento y construcción de modelos
    print("🔄 Procesando documentos...")
    processed_docs = [preprocess_text(doc) for doc in docs]
    inverted_index = build_inverted_index(processed_docs)
    
    # Construir modelos usando librerías
    tfidf_vectorizer, tfidf_matrix = build_tfidf_model(processed_docs)
    bm25_model = build_bm25_model(processed_docs)
    
    print("🎉 Sistema listo!")
    
    # Menú principal
    while True:
        print("\n" + "="*60)
        print("🔍 SISTEMA DE RECUPERACION DE INFORMACION")
        print("="*60)
        options = ["📋 Ver Informacion", "🔍 Búsqueda TF-IDF", "🎯 Búsqueda BM25", "📈 Evaluación", "❌ Salir"]
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        print("="*60)
        
        option = input("\nSelecciona una opción (1-5): ").strip()
        
        if option == "1":
            show_dataset_stats(docs, queries, qrels_dict, inverted_index)
        elif option == "2":
            search_interface(tfidf_vectorizer, tfidf_matrix, bm25_model, docs, doc_ids, "TF-IDF")
        elif option == "3":
            search_interface(tfidf_vectorizer, tfidf_matrix, bm25_model, docs, doc_ids, "BM25")
        elif option == "4":
            evaluation_interface(tfidf_vectorizer, tfidf_matrix, bm25_model, queries, qrels_dict, doc_id_to_index, processed_docs)
        elif option == "5":
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("\n❌ Opción no válida. Por favor selecciona 1-5.")
            input("📥 Presiona Enter para continuar...")

if __name__ == "__main__":
    main()