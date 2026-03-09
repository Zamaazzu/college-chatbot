from flask import Flask, request, jsonify, session
from flask_cors import CORS
from ml_logic.document_retriever import find_most_relevant_document
import pickle
import requests
import uuid

app = Flask(__name__)
app.secret_key = "college-chatbot-secret-key-change-in-production"
CORS(app, supports_credentials=True)

# ===============================
# Load ML model
# ===============================
model = pickle.load(open("models/intent_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))

# ===============================
# Topic Keywords
# ===============================
TOPIC_KEYWORDS = {
    "faculty":  ["faculty", "teacher", "hod", "professor"],
    "exam":     ["exam", "semester", "internal", "test"],
    "fees":     ["fee", "tuition", "deposit"],
    "library":  ["library", "reading", "books"],
    "event":    ["event", "fest", "celebration", "sports", "arts"],
    "class":    ["class", "lecture", "schedule", "timing"],
    "campus":   ["campus", "location", "building", "room"],
    "attendance": ["attendance", "absent", "present", "percentage"],
}

# ===============================
# Topic Detection
# ===============================
def detect_topic(text):
    text = text.lower()
    for topic, words in TOPIC_KEYWORDS.items():
        for word in words:
            if word in text:
                return topic
    return None


# ===============================
# Follow-up Query Enrichment (per-session)
# ===============================
def enrich_query(user_text):
    text = user_text.lower()

    follow_words = [
        "its", "their", "them", "that", "those",
        "what about", "and", "also", "another",
        "friday", "then", "next"
    ]

    topic_memory = session.get("topic_memory", {})
    last_topic   = session.get("last_topic", None)

    topic = detect_topic(text)

    if topic:
        topic_memory[topic] = text
        last_topic = topic
        session["topic_memory"] = topic_memory
        session["last_topic"]   = last_topic
        return text

    for word in follow_words:
        if word in text and last_topic:
            previous = topic_memory.get(last_topic)
            if previous:
                return previous + " " + text

    return text


# ===============================
# LLM Generation
# ===============================
def generate_with_llm(question, context, history):
    history_text = "\n".join(history[-6:])

    prompt = f"""
You are an intelligent AI college assistant that helps students with institutional information.

STRICT RULES:
- Use ONLY the information provided below.
- NEVER mention document names, PDF files, sources, or where the information came from.
- NEVER use words like "context", "document", "file", or "PDF".
- Answer naturally as if you already know the information.
- If the information does not clearly answer the question, say: "I don't have that information right now. Please contact the college administration office."

Response Style:
- Write 2–4 clear sentences.
- Be helpful and conversational.

Previous conversation:
{history_text}

Information:
{context}

Student question:
{question}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        result = response.json()
        if "response" in result:
            return result["response"].strip()
        return "Sorry, I couldn't generate a response."

    except requests.exceptions.Timeout:
        return "The AI response service timed out. Please try again."
    except Exception:
        return "The AI response service is currently unavailable."


# ===============================
# Chat API
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    # Per-session conversation history
    if "conversation_history" not in session:
        session["conversation_history"] = []
    if "topic_memory" not in session:
        session["topic_memory"] = {}
    if "last_topic" not in session:
        session["last_topic"] = None

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"response": "Invalid request"}), 400

    user_text  = data["message"].strip()
    lower_text = user_text.lower()

    # ===============================
    # Greetings
    # ===============================
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]

    if lower_text in greetings:
        response = (
            "Hello! I'm your AI college assistant. "
            "I can help with exams, fees, faculty information, "
            "library services, campus locations, and college events."
        )
        _save_history(user_text, response)
        return jsonify({"intent": "greeting", "response": response})

    # ===============================
    # Help Queries
    # ===============================
    help_queries = ["help", "what can you do", "how can you help me"]

    if lower_text in help_queries:
        response = (
            "I can answer questions about exam schedules, fee structure, "
            "faculty members, campus facilities, class timings, "
            "library services, and college events."
        )
        _save_history(user_text, response)
        return jsonify({"intent": "help", "response": response})

    # ===============================
    # Query Enrichment (per-session memory)
    # ===============================
    enriched_query = enrich_query(user_text)

    # ===============================
    # Intent Detection
    # ===============================
    vec   = vectorizer.transform([user_text])
    probs = model.predict_proba(vec)[0]

    if max(probs) < 0.2:
        intent = "unknown"
    else:
        intent = model.classes_[probs.argmax()]

    # ===============================
    # Response Logic — document retrieval for everything
    # ===============================
    doc_name, content = find_most_relevant_document(enriched_query)

    if content:
        print("\n--- Context sent to LLM ---")
        print(content[:500])
        print("--- End of Context ---\n")
        response = generate_with_llm(
            user_text,
            content,
            session["conversation_history"]
        )
    else:
        response = (
            "I couldn't find that information in the available documents. "
            "Please contact the college administration office."
        )

    # ===============================
    # Save Conversation
    # ===============================
    _save_history(user_text, response)

    return jsonify({"intent": intent, "response": response})


# ===============================
# Helper — save to session history
# ===============================
def _save_history(user_text, response):
    history = session.get("conversation_history", [])
    history.append("User: " + user_text)
    history.append("Bot: " + response)
    if len(history) > 20:
        history = history[-20:]
    session["conversation_history"] = history


# ===============================
# Run Server
# ===============================
if __name__ == "__main__":
    app.run(debug=True)