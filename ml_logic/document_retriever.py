import os
import pdfplumber
import re
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

nltk.download("punkt")

DOCS_PATH = "data/documents"

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# Global Cache
# ===============================
DOCUMENT_CHUNKS = []
DOCUMENT_NAMES = []
CHUNK_EMBEDDINGS = None
INITIALIZED = False


# ===============================
# Stopwords (NEW)
# ===============================
STOPWORDS = {
    "what","is","the","a","an","about","tell","me",
    "give","information","details","please","of",
    "in","on","for","to","and","with"
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
    filenames = []

    if not os.path.exists(DOCS_PATH):
        return documents, filenames

    for file in os.listdir(DOCS_PATH):

        if file.endswith(".pdf"):

            full_path = os.path.join(DOCS_PATH, file)
            text = ""

            with pdfplumber.open(full_path) as pdf:

                for page in pdf.pages:

                    extracted = page.extract_text()

                    if extracted:
                        text += extracted + "\n"

                    tables = page.extract_tables()

                    for table in tables:

                        for row in table:

                            row_text = " | ".join(
                                [cell.strip() if cell else "" for cell in row]
                            )

                            text += row_text + "\n"

            cleaned = clean_text(text)

            if len(cleaned) > 100:

                documents.append(cleaned)
                filenames.append(file)

    return documents, filenames


# ===============================
# Chunk text
# ===============================
def chunk_text(text, chunk_size=250, overlap=40):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):

        chunk = " ".join(words[i:i + chunk_size])

        if len(chunk) > 100:
            chunks.append(chunk)

    return chunks


# ===============================
# Query Expansion
# ===============================
def expand_query(query):

    query_lower = query.lower()

    expansion_map = {

        "library": "books reading study room digital library journal timing working hours",

        "exam": "examination semester test hall ticket rules timetable",

        "attendance": "minimum attendance percentage requirement rule",

        "fees": "tuition fee semester fee payment fee structure",

        "event": "college event fest program activity cultural technical fest",

        "faculty": "teacher professor department staff",

        "result": "exam result marks grade semester result",

        "admin": "administration office help desk contact"
    }

    expanded_query = query

    for key in expansion_map:

        if key in query_lower:
            expanded_query += " " + expansion_map[key]

    return expanded_query


# ===============================
# Initialize Document Cache
# ===============================
def initialize_documents():

    global DOCUMENT_CHUNKS, DOCUMENT_NAMES, CHUNK_EMBEDDINGS, INITIALIZED

    if INITIALIZED:
        return

    docs, names = load_documents()

    all_chunks = []
    all_doc_names = []

    for doc_text, doc_name in zip(docs, names):

        chunks = chunk_text(doc_text)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_doc_names.append(doc_name)

    if not all_chunks:
        return

    embeddings = embedding_model.encode(all_chunks)
    embeddings = normalize(embeddings)

    DOCUMENT_CHUNKS = all_chunks
    DOCUMENT_NAMES = all_doc_names
    CHUNK_EMBEDDINGS = embeddings

    INITIALIZED = True


# ===============================
# Semantic Retrieval
# ===============================
def find_most_relevant_document(query):

    global DOCUMENT_CHUNKS, DOCUMENT_NAMES, CHUNK_EMBEDDINGS

    initialize_documents()

    if not DOCUMENT_CHUNKS:
        return None, None

    # Query expansion
    query = expand_query(query)

    query_embedding = embedding_model.encode([query])
    query_embedding = normalize(query_embedding)[0]

    # remove stopwords
    query_words = [
        word for word in query.lower().split()
        if word not in STOPWORDS
    ]

    # Cosine similarity
    similarities = np.dot(CHUNK_EMBEDDINGS, query_embedding)

    # ===============================
    # Keyword Boosting (Improved)
    # ===============================
    for i, chunk in enumerate(DOCUMENT_CHUNKS):

        chunk_lower = chunk.lower()

        for word in query_words:

            if word in chunk_lower:
                similarities[i] += 0.08


    # ===============================
    # Select best chunks
    # ===============================
    top_k = min(5, len(similarities))
    top_indices = similarities.argsort()[-top_k:][::-1]

    selected_chunks = []
    doc_name = None
    used_chunks = set()

    for i in top_indices:

        if similarities[i] > 0.28:

            chunk = DOCUMENT_CHUNKS[i]

            if chunk not in used_chunks:

                selected_chunks.append(chunk)
                used_chunks.add(chunk)
                doc_name = DOCUMENT_NAMES[i]

    if not selected_chunks:
        return None, None

    combined_context = "\n\n".join(selected_chunks)

    return doc_name, combined_context