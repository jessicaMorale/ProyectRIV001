# Importa las librerías necesarias
import ir_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import math
from collections import Counter

# Descargar recursos necesarios de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Función que imprime una línea separadora
def print_separator():
    print("=" * 60)

# Función de preprocesamiento de texto
def preprocess_text(text):
    """
    Procesamiento básico del texto:
    - Tokenización
    - Normalización (minúsculas, remoción de puntuación)
    - Remoción de stopwords
    """
    if not text:
        return ""
    
    # NORMALIZACIÓN
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # TOKENIZACIÓN
    tokens = word_tokenize(text)
    
    # REMOCIÓN DE STOPWORDS
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(filtered_tokens)

# Función para construir índice invertido
def build_inverted_index(processed_docs):
    """
    Construye un índice invertido que almacena:
    - Para cada término: los documentos donde aparece y su frecuencia
    """
    print("🔄 Construyendo índice invertido...")
    inverted_index = {}
    
    for doc_id, doc in enumerate(processed_docs):
        if not doc.strip():
            continue
            
        # Tokenizar el documento procesado
        tokens = doc.split()
        
        # Contar frecuencia de cada término en este documento
        term_freq = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        
        # Agregar al índice invertido
        for term, freq in term_freq.items():
            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][doc_id] = freq
    
    print(f"✅ Índice invertido construido con {len(inverted_index)} términos únicos")
    return inverted_index

# Clase BM25 para recuperación
class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_count = len(documents)
        
        # Construir vocabulario y frecuencias de documentos
        self.vocab = set()
        self.doc_frequencies = {}
        
        for doc in documents:
            words = set(doc.split())
            self.vocab.update(words)
            for word in words:
                self.doc_frequencies[word] = self.doc_frequencies.get(word, 0) + 1
    
    def get_scores(self, query):
        scores = []
        query_words = query.split()
        
        for doc_idx, doc in enumerate(self.documents):
            doc_words = doc.split()
            doc_word_counts = Counter(doc_words)
            score = 0
            
            for word in query_words:
                if word in doc_word_counts:
                    # Frecuencia del término en el documento
                    tf = doc_word_counts[word]
                    
                    # Frecuencia inversa del documento
                    df = self.doc_frequencies.get(word, 0)
                    idf = math.log((self.doc_count - df + 0.5) / (df + 0.5))
                    
                    # Longitud del documento
                    doc_len = self.doc_lengths[doc_idx]
                    
                    # Fórmula BM25
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                    score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return scores
def show_index_stats(inverted_index, sample_terms=5):
    print(f"\n📊 ESTADÍSTICAS DEL ÍNDICE INVERTIDO:")
    print(f"   Total de términos: {len(inverted_index)}")
    
    # Términos más frecuentes (en más documentos)
    term_doc_counts = [(term, len(docs)) for term, docs in inverted_index.items()]
    term_doc_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   Términos en más documentos:")
    for term, doc_count in term_doc_counts[:sample_terms]:
        total_freq = sum(inverted_index[term].values())
        print(f"     '{term}': {doc_count} documentos, {total_freq} ocurrencias totales")
    
    print(f"   Términos en menos documentos:")
    for term, doc_count in term_doc_counts[-sample_terms:]:
        total_freq = sum(inverted_index[term].values())
        print(f"     '{term}': {doc_count} documentos, {total_freq} ocurrencias totales")

# Función para mostrar el menú principal
def show_main_menu():
    print("\n" + "="*60)
    print("🔍 SISTEMA DE BÚSQUEDA DE DOCUMENTOS")
    print("="*60)
    print("1. 📋 Ver estadísticas del dataset")
    print("2. 📊 Ver estadísticas del índice invertido")
    print("3. 🔍 Búsqueda con TF-IDF + Coseno")
    print("4. 🎯 Búsqueda con BM25")
    print("5. ❌ Salir")
    print("="*60)

# Función para mostrar estadísticas del dataset
def show_dataset_stats(docs, vectorizer, tfidf_matrix):
    print(f"\n📊 ESTADÍSTICAS DEL DATASET:")
    print_separator()
    print(f"   Total de documentos: {len(docs)}")
    print(f"   Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} términos")
    print(f"   Matriz TF-IDF: {tfidf_matrix.shape}")
    input("\n📥 Presiona Enter para continuar...")

# Interfaz de búsqueda con TF-IDF
def search_interface_tfidf(vectorizer, tfidf_matrix, docs, doc_ids):
    while True:
        print("\n🔍 BÚSQUEDA CON TF-IDF + COSENO")
        print_separator()
        print("Escribe tu consulta (o 'atras' para volver al menú principal)")
        query = input("\n> ").strip()
        

        if query.lower() == 'atras':
            break
        if not query:
            print("❌ Por favor ingresa una consulta válida.")
            continue

        print(f"\n🔄 Procesando consulta: '{query}'...")
        
        # Preprocesar la consulta igual que los documentos
        processed_query = preprocess_text(query)
        if not processed_query.strip():
            print("❌ La consulta no contiene términos válidos después del preprocesamiento.")
            continue
            
        query_vector = vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        relevant_docs = [(i, sim) for i, sim in enumerate(similarities) if sim > 0]
        relevant_docs.sort(key=lambda x: x[1], reverse=True)

        if not relevant_docs:
            print("❌ No se encontraron documentos relevantes.")
            continue

        print(f"\n📋 RESULTADOS TF-IDF PARA: '{query}'")
        print(f"Se encontraron {len(relevant_docs)} documentos relevantes")
        print_separator()

        for rank, (doc_id, sim) in enumerate(relevant_docs[:5], start=1):
            print(f"\n🔸 RESULTADO #{rank}")
            print(f"   Similitud Coseno: {sim:.4f}")
            print(f"   ID Documento: {doc_ids[doc_id]}")
            print(f"   Contenido: {docs[doc_id][:200].replace('\n', ' ')}...")
            if rank < 5 and rank < len(relevant_docs):
                print("   " + "─" * 50)

        if len(relevant_docs) > 5:
            print(f"\n   📄 ... y {len(relevant_docs) - 5} documentos más")

        input("\n📥 Presiona Enter para continuar...")

# Interfaz de búsqueda con BM25
def search_interface_bm25(bm25_model, docs, doc_ids):
    while True:
        print("\n🎯 BÚSQUEDA CON BM25")
        print_separator()
        print("Escribe tu consulta (o 'atras' para volver al menú principal)")
        query = input("\n> ").strip()
        

        if query.lower() == 'atras':
            break
        if not query:
            print("❌ Por favor ingresa una consulta válida.")
            continue

        print(f"\n🔄 Procesando consulta: '{query}'...")
        
        # Preprocesar la consulta igual que los documentos
        processed_query = preprocess_text(query)
        if not processed_query.strip():
            print("❌ La consulta no contiene términos válidos después del preprocesamiento.")
            continue
            
        scores = bm25_model.get_scores(processed_query)
        relevant_docs = [(i, score) for i, score in enumerate(scores) if score > 0]
        relevant_docs.sort(key=lambda x: x[1], reverse=True)

        if not relevant_docs:
            print("❌ No se encontraron documentos relevantes.")
            continue

        print(f"\n📋 RESULTADOS BM25 PARA: '{query}'")
        print(f"Se encontraron {len(relevant_docs)} documentos relevantes")
        print_separator()

        for rank, (doc_id, score) in enumerate(relevant_docs[:5], start=1):
            print(f"\n🔸 RESULTADO #{rank}")
            print(f"   Puntuación BM25: {score:.4f}")
            print(f"   ID Documento: {doc_ids[doc_id]}")
            print(f"   Contenido: {docs[doc_id][:200].replace('\n', ' ')}...")
            if rank < 5 and rank < len(relevant_docs):
                print("   " + "─" * 50)

        if len(relevant_docs) > 5:
            print(f"\n   📄 ... y {len(relevant_docs) - 5} documentos más")

        input("\n📥 Presiona Enter para continuar...")

# Función principal del menú
def main_menu(docs, doc_ids, vectorizer, tfidf_matrix, inverted_index, bm25_model):
    while True:
        show_main_menu()
        option = input("\nSelecciona una opción (1-5): ").strip()
        
        if option == "1":
            show_dataset_stats(docs, vectorizer, tfidf_matrix)
        elif option == "2":
            show_index_stats(inverted_index)
            input("\n📥 Presiona Enter para continuar...")
        elif option == "3":
            search_interface_tfidf(vectorizer, tfidf_matrix, docs, doc_ids)
        elif option == "4":
            search_interface_bm25(bm25_model, docs, doc_ids)
        elif option == "5":
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("\n❌ Opción no válida. Por favor selecciona 1-5.")
            input("📥 Presiona Enter para continuar...")

# ================= EJECUCIÓN ===================

# Carga del dataset BEIR CQADupStack programmers
print("🔄 Cargando dataset BEIR CQADupStack programmers...")
dataset = ir_datasets.load("beir/cqadupstack/programmers")

# Extraer documentos del dataset
docs = []
doc_ids = []

print("🔄 Extrayendo documentos...")
for doc in dataset.docs_iter():
    combined_text = ""
    if hasattr(doc, 'title') and doc.title:
        combined_text += doc.title + " "
    if hasattr(doc, 'text') and doc.text:
        combined_text += doc.text
    
    docs.append(combined_text.strip())
    doc_ids.append(doc.doc_id)

print(f"✅ Dataset cargado: {len(docs)} documentos.")

# Aplicar preprocesamiento a los documentos
print("🔄 Aplicando preprocesamiento a los documentos...")
processed_docs = [preprocess_text(doc) for doc in docs]

# Construir índice invertido
inverted_index = build_inverted_index(processed_docs)

# Vectorización con preprocesamiento
print("🔄 Vectorizando documentos...")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_docs)
print("✅ Vectorización completada.")

# Construir modelo BM25
print("🔄 Construyendo modelo BM25...")
bm25_model = BM25(processed_docs)
print("✅ Modelo BM25 construido.")

print("\n🎉 Sistema listo!")

# Ejecutar el menú principal
main_menu(docs, doc_ids, vectorizer, tfidf_matrix, inverted_index, bm25_model)
