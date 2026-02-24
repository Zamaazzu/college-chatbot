from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_logic.document_retriever import find_most_relevant_document
import pickle
import requests

app = Flask(__name__)
CORS(app)

# ===============================
# Load trained ML model
# ===============================
model = pickle.load(open("models/intent_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))

# ===============================
# Conversation Memory
# ===============================
conversation_history = []

# Structured entity memory (generic + expandable)
entity_memory = {
    "department": None,
    "semester": None,
    "course": None
}

# ===============================
# Entity Extraction (Generic)
# ===============================
def extract_entities(text):
    text = text.lower()
    entities = {}

    departments = ["cse", "eee", "mca", "electrical", "computer science"]
    semesters = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]

    for dept in departments:
        if dept in text:
            entities["department"] = dept

    for sem in semesters:
        if sem in text:
            entities["semester"] = sem

    return entities


# ===============================
# Inject Entity Memory Into Query
# ===============================
def enrich_query_with_memory(user_text):
    enriched_query = user_text.lower()

    for key, value in entity_memory.items():
        if value and value not in enriched_query:
            enriched_query += f" {value}"

    return enriched_query


# ===============================
# LLM Generation Function
# ===============================
def generate_with_llm(question, context):
    history_text = "\n".join(conversation_history[-6:])

    prompt = f"""
You are CollegeBot, a smart and helpful AI assistant for a college.

Your role:
- Use the official information below as the source of truth.
- You may rephrase, structure, and present the information naturally.
- Do NOT mention "context" or "provided information".
- If the answer is not found, say:
  "I do not have that information in the official documents."
- Provide a helpful, slightly detailed response (2–4 sentences when possible).
- Sound natural and conversational.

Conversation so far:
{history_text}

Official Information:
{context}

User Question:
{question}

Respond as CollegeBot:
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7
            }
        )
        return response.json()["response"].strip()

    except Exception:
        return "LLM service is currently unavailable."

# ===============================
# API Route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    global conversation_history
    global entity_memory

    data = request.get_json()
    user_text = data["message"].strip()

    # ===============================
    # Greeting Shortcut
    # ===============================
    if user_text.lower() in ["hi", "hello", "hey"]:
        response = "Hello! I’m your college assistant. How can I help you?"

        conversation_history.append("User: " + user_text)
        conversation_history.append("Bot: " + response)

        return jsonify({"intent": "greeting", "response": response})

    # ===============================
    # Update Entity Memory
    # ===============================
    new_entities = extract_entities(user_text)
    for key, value in new_entities.items():
        entity_memory[key] = value

    # ===============================
    # Intent Detection (Optional)
    # ===============================
    vec = vectorizer.transform([user_text])
    probs = model.predict_proba(vec)[0]
    max_prob = max(probs)

    if max_prob < 0.2:
        intent = "unknown"
    else:
        intent = model.classes_[probs.argmax()]

    # ===============================
    # Enrich Query Using Memory
    # ===============================
    enriched_query = enrich_query_with_memory(user_text)

    # ===============================
    # Retrieval + LLM (Unified System)
    # ===============================
    doc_name, content = find_most_relevant_document(enriched_query)

    if content:
        print("\n--- Context sent to LLM ---")
        print(content)
        print("--- End of Context ---\n")

        response = generate_with_llm(enriched_query, content)
    else:
        response = "I do not have that information in the official documents."

    # ===============================
    # Save Conversation Memory
    # ===============================
    conversation_history.append("User: " + user_text)
    conversation_history.append("Bot: " + response)

    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    return jsonify({
        "intent": intent,
        "response": response
    })


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)