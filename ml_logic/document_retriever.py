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
DOCUMENT_CHUNKS  = []
DOCUMENT_NAMES   = []
CHUNK_EMBEDDINGS = None
TFIDF_MATRIX     = None
TFIDF_VECTORIZER = None
INITIALIZED      = False
_init_lock       = threading.Lock()

# ===============================
# Stopwords
# ===============================
STOPWORDS = {
    "what", "is", "the", "a", "an", "about", "tell", "me",
    "give", "information", "details", "please", "of",
    "in", "on", "for", "to", "and", "with"
}

# ===============================
# High-value domain words — get a stronger boost
# ===============================
HIGH_VALUE_WORDS = {
    "fee", "fees", "tuition", "exam", "examination", "semester",
    "attendance", "faculty", "professor", "hod", "library",
    "schedule", "timetable", "result", "marks", "placement",
    "hostel", "event", "fest", "campus", "admission"
}

# ===============================
# PDF Boilerplate patterns to strip
# Repeated headers/footers pollute chunks and waste embedding space
# ===============================
BOILERPLATE_PATTERNS = [
    r'page\s+\d+\s+(of\s+\d+)?',   # "Page 1 of 10"
    r'confidential',
    r'www\.\S+',                     # website URLs
    r'\S+@\S+\.\S+',                 # email addresses
    r'tel\s*:\s*[\d\s\-]+',          # phone numbers
    r'^\s*\d+\s*$',                  # lone page numbers
]
_boilerplate_re = re.compile(
    "|".join(BOILERPLATE_PATTERNS), re.IGNORECASE | re.MULTILINE
)

# ===============================
# Text Cleaning
# ===============================
def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode()
    text = _boilerplate_re.sub(" ", text)                    # strip boilerplate
    text = re.sub(r'[^a-zA-Z0-9Rs./\-:,\s]', '', text)
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

    for file in sorted(os.listdir(DOCS_PATH)):
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
# Section-aware chunking
# Splits on natural boundaries (headings, blank lines, table rows)
# before falling back to word-count chunks — keeps content intact
# ===============================
def chunk_text(text, chunk_size=200, overlap=40):

    # Split on double newlines or lines that look like section headings
    # (short lines, all-caps, or ending with ":")
    section_re = re.compile(
        r'\n{2,}'                        # blank lines
        r'|(?<=[.!?])\s*\n'             # sentence-ending newlines
        r'|^[A-Z][A-Z\s]{3,}:\s*\n'    # ALL-CAPS headings
        r'|^.{0,60}:\s*\n',             # short heading lines ending with ":"
        re.MULTILINE
    )

    raw_sections = section_re.split(text)
    sections     = [s.strip() for s in raw_sections if len(s.strip()) > 60]

    chunks = []
    buffer_words = []

    for section in sections:
        words = section.split()

        # If section fits within one chunk, keep it whole
        if len(words) <= chunk_size:
            if len(buffer_words) + len(words) <= chunk_size:
                buffer_words.extend(words)
            else:
                if buffer_words:
                    chunks.append(" ".join(buffer_words))
                buffer_words = words
        else:
            # Flush buffer first
            if buffer_words:
                chunks.append(" ".join(buffer_words))
                buffer_words = []
            # Then word-count split the long section
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i : i + chunk_size])
                if len(chunk) > 80:
                    chunks.append(chunk)

    if buffer_words:
        chunks.append(" ".join(buffer_words))

    return [c for c in chunks if len(c) > 80]


# ===============================
# Query Expansion
# ===============================
def expand_query(query):
    query_lower = query.lower()

    # All club names expand to the full clubs string so Llama3
    # always sees ALL clubs in context, not just the one mentioned
    ALL_CLUBS = (
        "IEEE IEDC TinkerHub GDC Google Developer Club forums student branch "
        "technical workshop coding hackathon innovation entrepreneurship community "
        "event program activity club"
    )

    expansion_map = {
        "library":          "books reading study room digital library journal timing working hours",
        "exam":             "examination semester test hall ticket rules timetable",
        "attendance":       "minimum attendance percentage requirement rule",
        "fees":             "tuition fee semester fee payment fee structure",
        "event":            "college event fest program activity cultural technical onam christmas sports arts " + ALL_CLUBS,
        "faculty":          "teacher professor department staff hod",
        "result":           "exam result marks grade semester result",
        "admin":            "administration office help desk contact",
        "hostel":           "hostel accommodation room warden rules",
        "placement":        "placement recruitment company drive internship",
        # Any single club name mentioned → pull all clubs into context
        "ieee":             ALL_CLUBS,
        "iedc":             ALL_CLUBS,
        "tinkerhub":        ALL_CLUBS,
        "gdc":              ALL_CLUBS,
        "google developer": ALL_CLUBS,
        "forum":            ALL_CLUBS,
        "club":             ALL_CLUBS,
        "workshop":         ALL_CLUBS,
        "hackathon":        ALL_CLUBS,
        "fest":             "college event fest program activity cultural onam christmas sports arts " + ALL_CLUBS,
        "onam":             "college event cultural celebration onam festival",
        "christmas":        "college event cultural celebration christmas festival",
        "sports":           "annual sports meet athletics football basketball track field",
        "arts":             "arts festival dance music drama painting literary even semester",
    }

    expanded = query
    for key, expansion in expansion_map.items():
        if key in query_lower:
            expanded += " " + expansion

    return expanded


# ===============================
# Chunk deduplication helper
# Skips a new chunk if it shares >60% words with any already-selected chunk
# Prevents sending near-duplicate overlapping content to Llama3
# ===============================
def _is_duplicate(chunk, selected_chunks, threshold=0.6):
    chunk_words = set(chunk.lower().split())
    for existing in selected_chunks:
        existing_words = set(existing.lower().split())
        if not chunk_words or not existing_words:
            continue
        overlap = len(chunk_words & existing_words) / min(len(chunk_words), len(existing_words))
        if overlap > threshold:
            return True
    return False


# ===============================
# Initialize Document Cache (thread-safe, runs once)
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

        # Semantic embeddings
        embeddings = embedding_model.encode(all_chunks, show_progress_bar=False)
        embeddings = normalize(embeddings)

        # TF-IDF matrix for hybrid scoring
        tfidf_vec    = TfidfVectorizer(stop_words="english", max_features=10000)
        tfidf_matrix = tfidf_vec.fit_transform(all_chunks)

        DOCUMENT_CHUNKS  = all_chunks
        DOCUMENT_NAMES   = all_doc_names
        CHUNK_EMBEDDINGS = embeddings
        TFIDF_MATRIX     = tfidf_matrix
        TFIDF_VECTORIZER = tfidf_vec

        INITIALIZED = True
        print(f"[INFO] Loaded {len(all_chunks)} chunks from {len(docs)} documents.")


# ===============================
# Hybrid Retrieval
# Returns (doc_name, combined_context, used_fallback)
# ===============================
def find_most_relevant_document(query, top_k=5, threshold=0.25):

    initialize_documents()

    if not DOCUMENT_CHUNKS:
        return None, None, False

    # Query expansion
    query = expand_query(query)

    # --- Semantic similarity ---
    q_embedding     = embedding_model.encode([query], show_progress_bar=False)
    q_embedding     = normalize(q_embedding)[0]
    semantic_scores = np.dot(CHUNK_EMBEDDINGS, q_embedding)

    # --- TF-IDF similarity ---
    q_tfidf      = TFIDF_VECTORIZER.transform([query])
    tfidf_scores = (TFIDF_MATRIX @ q_tfidf.T).toarray().flatten()
    tfidf_max    = tfidf_scores.max()
    tfidf_scores = tfidf_scores / (tfidf_max + 1e-9)   # safe normalisation

    # --- Hybrid score ---
    combined_scores = 0.6 * semantic_scores + 0.4 * tfidf_scores

    # --- Keyword boosting ---
    query_words = [w for w in query.lower().split() if w not in STOPWORDS]

    for i, chunk in enumerate(DOCUMENT_CHUNKS):
        chunk_lower  = chunk.lower()
        doc_name_low = DOCUMENT_NAMES[i].lower()

        for word in query_words:
            if word in chunk_lower:
                # High-value domain words get a stronger boost
                boost = 0.12 if word in HIGH_VALUE_WORDS else 0.06
                combined_scores[i] += boost

            # Filename match bonus — e.g. "fee" in "fee_structure.pdf"
            if word in doc_name_low:
                combined_scores[i] += 0.10

    # --- Select top-k, deduplicated chunks ---
    top_k_actual = min(top_k, len(combined_scores))
    top_indices  = combined_scores.argsort()[-top_k_actual:][::-1]

    selected_chunks = []
    used_fallback   = False

    for i in top_indices:
        if combined_scores[i] > threshold:
            chunk = DOCUMENT_CHUNKS[i]
            if not _is_duplicate(chunk, selected_chunks):
                selected_chunks.append(chunk)

    # --- Fallback: return best chunk if nothing clears threshold ---
    if not selected_chunks:
        best_idx = combined_scores.argmax()
        if combined_scores[best_idx] > 0.10:
            selected_chunks = [DOCUMENT_CHUNKS[best_idx]]
            top_indices     = [best_idx]
            used_fallback   = True
        else:
            return None, None, False

    doc_name         = DOCUMENT_NAMES[top_indices[0]]
    combined_context = "\n\n".join(selected_chunks)

    return doc_name, combined_context, used_fallback