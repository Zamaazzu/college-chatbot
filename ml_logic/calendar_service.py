import os
import re
import numpy as np
from datetime import datetime
import pdfplumber
from sentence_transformers import SentenceTransformer

DOCS_PATH = "data/documents"

class CalendarService:
    def __init__(self):
        self.events = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.event_embeddings = None
        self.load_calendars()
        self.create_embeddings()

    # 1️⃣ Load all calendar PDFs
    def load_calendars(self):
        for file in os.listdir(DOCS_PATH):
            if file.endswith(".pdf") and "calendar" in file.lower():
                full_path = os.path.join(DOCS_PATH, file)
                self.extract_events(full_path, file)

    # 2️⃣ Extract event + date
    def extract_events(self, pdf_path, source):
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                lines = text.split("\n")

                for line in lines:
                    match = re.search(
                        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                        line
                    )

                    if match:
                        date_str = match.group(0)
                        parsed_date = self.parse_date(date_str)

                        if parsed_date:
                            self.events.append({
                                "event": line.strip(),
                                "date": parsed_date,
                                "source": source
                            })

    # 3️⃣ Convert string to datetime
    def parse_date(self, date_str):
        formats = [
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d %B %Y",
            "%d %b %Y"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        return None

    # 4️⃣ Create embeddings once
    def create_embeddings(self):
        if not self.events:
            return

        texts = [e["event"] for e in self.events]
        self.event_embeddings = self.model.encode(texts)

    # 5️⃣ Search most relevant event
    def search_event(self, query):
        if not self.events or self.event_embeddings is None:
            return None

        query_embedding = self.model.encode([query])
        similarities = np.dot(self.event_embeddings, query_embedding.T).flatten()

        best_index = similarities.argmax()
        best_score = similarities[best_index]

        if best_score < 0.4:
            return None

        return self.events[best_index]