import os
import re
import threading
import numpy as np
import nltk
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt", quiet=True)

DOCS_PATH = "data/documents"

# Load embedding model once at import time
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# Global Cache
# ===============================
DOCUMENT_CHUNKS    = []
DOCUMENT_NAMES     = []
CHUNK_EMBEDDINGS   = None   # shape: (n_chunks, embedding_dim)
TFIDF_MATRIX       = None   # shape: (n_chunks, vocab)
TFIDF_VECTORIZER   = None
INITIALIZED        = False
_init_lock         = threading.Lock()   # thread-safe one-time init


# ===============================
# Stopwords
# ===============================
STOPWORDS = {
    "what", "is", "the", "a", "an", "about", "tell", "me",
    "give", "information", "details", "please", "of",
    "in", "on", "for", "to", "and", "with"
}


# ===============================
# Text Cleaning
# ===============================
def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r'[^a-zA-Z0-9₹Rs./\-:,\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ===============================
# Load PDFs
# ===============================
def load_documents():
    documents = []
    filenames  = []

    if not os.path.exists(DOCS_PATH):
        print(f"[WARNING] Documents path not found: {DOCS_PATH}")
        return documents, filenames

    for file in os.listdir(DOCS_PATH):
        if not file.endswith(".pdf"):
            continue

        full_path = os.path.join(DOCS_PATH, file)
        text = ""

        try:
            with pdfplumber.open(full_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                    for table in page.extract_tables():
                        for row in table:
                            row_text = " | ".join(
                                cell.strip() if cell else "" for cell in row
                            )
                            text += row_text + "\n"

        except Exception as e:
            print(f"[ERROR] Could not read {file}: {e}")
            continue

        cleaned = clean_text(text)
        if len(cleaned) > 100:
            documents.append(cleaned)
            filenames.append(file)

    return documents, filenames


# ===============================
# Chunk text
# (smaller chunks for table-heavy docs)
# ===============================
def chunk_text(text, chunk_size=200, overlap=40):
    words  = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk) > 80:
            chunks.append(chunk)

    return chunks


# ===============================
# Query Expansion
# ===============================
def expand_query(query):
    query_lower = query.lower()

    expansion_map = {
        "library":    "books reading study room digital library journal timing working hours",
        "exam":       "examination semester test hall ticket rules timetable",
        "attendance": "minimum attendance percentage requirement rule",
        "fees":       "tuition fee semester fee payment fee structure",
        "event":      "college event fest program activity cultural technical",
        "faculty":    "teacher professor department staff hod",
        "result":     "exam result marks grade semester result",
        "admin":      "administration office help desk contact",
        "hostel":     "hostel accommodation room warden rules",
        "placement":  "placement recruitment company drive internship",
    }

    expanded = query
    for key, expansion in expansion_map.items():
        if key in query_lower:
            expanded += " " + expansion

    return expanded


# ===============================
# Initialize Document Cache (thread-safe)
# ===============================
def initialize_documents():
    global DOCUMENT_CHUNKS, DOCUMENT_NAMES
    global CHUNK_EMBEDDINGS, TFIDF_MATRIX, TFIDF_VECTORIZER, INITIALIZED

    with _init_lock:
        if INITIALIZED:
            return

        docs, names = load_documents()

        if not docs:
            print("[WARNING] No documents loaded.")
            INITIALIZED = True
            return

        all_chunks    = []
        all_doc_names = []

        for doc_text, doc_name in zip(docs, names):
            chunks = chunk_text(doc_text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_doc_names.append(doc_name)

        if not all_chunks:
            INITIALIZED = True
            return

        # --- Semantic embeddings ---
        embeddings = embedding_model.encode(all_chunks, show_progress_bar=False)
        embeddings = normalize(embeddings)

        # --- TF-IDF matrix for hybrid scoring ---
        tfidf_vec    = TfidfVectorizer(stop_words="english", max_features=10000)
        tfidf_matrix = tfidf_vec.fit_transform(all_chunks)

        DOCUMENT_CHUNKS   = all_chunks
        DOCUMENT_NAMES    = all_doc_names
        CHUNK_EMBEDDINGS  = embeddings
        TFIDF_MATRIX      = tfidf_matrix
        TFIDF_VECTORIZER  = tfidf_vec

        INITIALIZED = True
        print(f"[INFO] Initialized {len(all_chunks)} chunks from {len(docs)} documents.")


# ===============================
# Semantic + TF-IDF Hybrid Retrieval
# ===============================
def find_most_relevant_document(query, top_k=5, threshold=0.25):
    """
    Returns (doc_name, combined_context) for the best matching chunks.
    Falls back to the single best chunk if nothing clears the threshold.
    """
    initialize_documents()

    if not DOCUMENT_CHUNKS:
        return None, None

    # --- Expand query ---
    query = expand_query(query)

    # --- Semantic similarity ---
    q_embedding = embedding_model.encode([query], show_progress_bar=False)
    q_embedding = normalize(q_embedding)[0]
    semantic_scores = np.dot(CHUNK_EMBEDDINGS, q_embedding)   # (n_chunks,)

    # --- TF-IDF similarity ---
    q_tfidf  = TFIDF_VECTORIZER.transform([query])
    tfidf_scores = (TFIDF_MATRIX @ q_tfidf.T).toarray().flatten()

    # Normalise TF-IDF scores to [0, 1] range before blending
    tfidf_max = tfidf_scores.max()
    if tfidf_max > 0:
        tfidf_scores = tfidf_scores / tfidf_max

    # --- Hybrid score (60% semantic, 40% TF-IDF) ---
    combined_scores = 0.6 * semantic_scores + 0.4 * tfidf_scores

    # --- Keyword boosting ---
    query_words = [w for w in query.lower().split() if w not in STOPWORDS]

    for i, chunk in enumerate(DOCUMENT_CHUNKS):
        chunk_lower = chunk.lower()
        for word in query_words:
            if word in chunk_lower:
                combined_scores[i] += 0.06

    # --- Select top-k chunks ---
    top_k_actual = min(top_k, len(combined_scores))
    top_indices  = combined_scores.argsort()[-top_k_actual:][::-1]

    selected_chunks = []
    used_chunks     = set()

    for i in top_indices:
        if combined_scores[i] > threshold:
            chunk = DOCUMENT_CHUNKS[i]
            if chunk not in used_chunks:
                selected_chunks.append(chunk)
                used_chunks.add(chunk)

    # --- Fallback: always return best chunk if nothing clears threshold ---
    if not selected_chunks:
        best_idx = combined_scores.argmax()
        # Only use fallback if there's at least a weak signal
        if combined_scores[best_idx] > 0.10:
            return DOCUMENT_NAMES[best_idx], DOCUMENT_CHUNKS[best_idx]
        return None, None

    # doc_name = document of the highest-scoring chunk (top_indices[0])
    doc_name = DOCUMENT_NAMES[top_indices[0]]
    combined_context = "\n\n".join(selected_chunks)

    return doc_name, combined_context