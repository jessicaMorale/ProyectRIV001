# Importa las librerías necesarias
import ir_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Función que imprime una línea separadora
def print_separator():
    print("=" * 60)

# Interfaz de búsqueda que no aplica preprocesamiento
def search_interface_no_preprocess(vectorizer, tfidf_matrix, docs, doc_ids):
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
            print(f"   ID Documento: {doc_ids[doc_id]}")
            print(f"   Contenido: {docs[doc_id][:200].replace('\n', ' ')}...")
            if rank < 5 and rank < len(relevant_docs):
                print("   " + "─" * 50)

        if len(relevant_docs) > 5:
            print(f"\n   📄 ... y {len(relevant_docs) - 5} documentos más")

        input("\n📥 Presiona Enter para continuar...")

# ================= EJECUCIÓN ===================

# Carga del dataset BEIR CQADupStack programmers
print("🔄 Cargando dataset BEIR CQADupStack programmers...")
dataset = ir_datasets.load("beir/cqadupstack/programmers")

# Extraer documentos del dataset
docs = []
doc_ids = []

print("🔄 Extrayendo documentos...")
for doc in dataset.docs_iter():
    # Combinar título y texto del documento
    combined_text = ""
    if hasattr(doc, 'title') and doc.title:
        combined_text += doc.title + " "
    if hasattr(doc, 'text') and doc.text:
        combined_text += doc.text
    
    docs.append(combined_text.strip())
    doc_ids.append(doc.doc_id)

print(f"✅ Dataset cargado: {len(docs)} documentos.")

# Vectorización sin preprocesamiento personalizado
print("🔄 Vectorizando documentos...")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)
print("✅ Vectorización completada.")

# Mostrar algunas estadísticas del dataset
print(f"\n📊 ESTADÍSTICAS DEL DATASET:")
print(f"   Total de documentos: {len(docs)}")
print(f"   Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} términos")
print(f"   Matriz TF-IDF: {tfidf_matrix.shape}")

# Ejecutar la interfaz
search_interface_no_preprocess(vectorizer, tfidf_matrix, docs, doc_ids)
