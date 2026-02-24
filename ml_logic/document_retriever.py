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
# Text Cleaning
# ===============================
def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r'[^a-zA-Z0-9₹Rs./\-:,\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ===============================
# Load All PDFs
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

                    # Extract normal text
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                    # Extract tables
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
# Chunking
# ===============================
def chunk_text(text, chunk_size=450):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 100:
            chunks.append(chunk)

    return chunks


# ===============================
# Semantic Retrieval
# ===============================
def find_most_relevant_document(query):
    docs, names = load_documents()

    if not docs:
        return None, None

    best_score = -1
    best_doc_name = None
    best_context = None

    # Encode query
    query_embedding = embedding_model.encode([query])
    query_embedding = normalize(query_embedding)[0]

    for doc_text, doc_name in zip(docs, names):

        chunks = chunk_text(doc_text)
        if not chunks:
            continue

        # Encode chunks
        chunk_embeddings = embedding_model.encode(chunks)
        chunk_embeddings = normalize(chunk_embeddings)

        # Cosine similarity
        similarities = np.dot(chunk_embeddings, query_embedding)

        # Top 3 most similar chunks
        top_k = min(3, len(similarities))
        top_indices = similarities.argsort()[-top_k:][::-1]

        combined_context = " ".join([chunks[i] for i in top_indices])
        top_score = similarities[top_indices[0]]

        if top_score > best_score:
            best_score = top_score
            best_doc_name = doc_name
            best_context = combined_context

    

    return best_doc_name, best_context