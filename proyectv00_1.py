# Importa el vectorizador TF-IDF y la función para calcular similitud del coseno
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Función que imprime una línea separadora
def print_separator():
    print("=" * 60)

# Interfaz de búsqueda que no aplica preprocesamiento
def search_interface_no_preprocess(vectorizer, tfidf_matrix, docs):
    while True:
        print("\n🔍 BÚSQUEDA DE DOCUMENTOS ")
        print_separator()
        print("Escribe tu consulta (o 'menu' para volver al menú principal)")
        query = input("\n> ").strip()

        if query.lower() == 'menu':
            break
        if not query:
            print("❌ Por favor ingresa una consulta válida.")
            continue

        print(f"\n🔄 Procesando consulta: '{query}'...")
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        relevant_docs = [(i, sim) for i, sim in enumerate(similarities) if sim > 0]
        relevant_docs.sort(key=lambda x: x[1], reverse=True)

        if not relevant_docs:
            print("❌ No se encontraron documentos relevantes.")
            continue

        print(f"\n📋 RESULTADOS PARA: '{query}'")
        print(f"Se encontraron {len(relevant_docs)} documentos relevantes")
        print_separator()

        for rank, (doc_id, sim) in enumerate(relevant_docs[:5], start=1):
            print(f"\n🔸 RESULTADO #{rank}")
            print(f"   Similitud: {sim:.4f}")
            print(f"   ID Documento: {doc_id}")
            print(f"   Contenido: {docs[doc_id][:200].replace('\n', ' ')}...")
            if rank < 5 and rank < len(relevant_docs):
                print("   " + "─" * 50)

        if len(relevant_docs) > 5:
            print(f"\n   📄 ... y {len(relevant_docs) - 5} documentos más")

        input("\n📥 Presiona Enter para continuar...")

# ================= EJECUCIÓN ===================

# Carga del dataset completo sin encabezados, pies ni citas
print("🔄 Cargando dataset 20 Newsgroups...")
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
docs = newsgroups.data
print(f"✅ Dataset cargado: {len(docs)} documentos.")

# Vectorización sin preprocesamiento personalizado
print("🔄 Vectorizando documentos...")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)
print("✅ Vectorización completada.")

# Ejecutar la interfaz
search_interface_no_preprocess(vectorizer, tfidf_matrix, docs)

