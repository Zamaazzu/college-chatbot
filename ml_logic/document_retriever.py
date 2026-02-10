import os
import pdfplumber
import re
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

nltk.download("punkt")

DOCS_PATH = "data/documents"

# Load embedding model once (semantic retrieval)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r'[^a-zA-Z0-9₹Rs./\-:,\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_documents():
    documents = []
    filenames = []

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
            documents.append(cleaned)
            filenames.append(file)

    return documents, filenames


def chunk_text(text, chunk_size=450):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 100:
            chunks.append(chunk)

    return chunks


def find_most_relevant_document(query):
    docs, names = load_documents()

    if not docs:
        return None, None

    best_score = -1
    best_doc_name = None
    best_chunk = None

    query_embedding = embedding_model.encode([query])[0]

    for doc_text, doc_name in zip(docs, names):

        chunks = chunk_text(doc_text)
        if not chunks:
            continue

        chunk_embeddings = embedding_model.encode(chunks)

        # Cosine similarity using dot product
        similarities = np.dot(chunk_embeddings, query_embedding)

        max_index = similarities.argmax()
        score = similarities[max_index]

        if score > best_score:
            best_score = score
            best_doc_name = doc_name
            best_chunk = chunks[max_index]

    if best_score < 0.25:
        return None, None

    return best_doc_name, best_chunk
