from flask import Flask, request, jsonify, session
from flask_cors import CORS
from ml_logic.document_retriever import find_most_relevant_document
import pickle
import requests
import os

app = Flask(__name__)
app.secret_key = "college-chatbot-secret-key-change-in-production"
CORS(app, supports_credentials=True)

# ===============================
# Load ML model
# ===============================
model      = pickle.load(open("models/intent_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf.pkl",        "rb"))

# ===============================
# Ensure log directory exists
# ===============================
os.makedirs("logs", exist_ok=True)

# ===============================
# Topic Keywords
# ===============================
TOPIC_KEYWORDS = {
    "faculty":    ["faculty", "teacher", "hod", "professor"],
    "exam":       ["exam", "semester", "internal", "test"],
    "fees":       ["fee", "tuition", "deposit"],
    "library":    ["library", "reading", "books"],
    "event":      ["event", "fest", "celebration", "sports", "arts",
                   "ieee", "iedc", "tinkerhub", "gdc", "onam", "christmas"],
    "class":      ["class", "lecture", "schedule", "timing"],
    "campus":     ["campus", "location", "building", "room"],
    "attendance": ["attendance", "absent", "present", "percentage"],
    "hostel":     ["hostel", "accommodation", "warden", "room"],
    "placement":  ["placement", "recruitment", "company", "internship"],
    "result":     ["result", "marks", "grade", "pass", "fail"],
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
        "friday", "then", "next", "when", "where", "how"
    ]

    topic_memory = session.get("topic_memory", {})
    last_topic   = session.get("last_topic",   None)

    topic = detect_topic(text)

    if topic:
        topic_memory[topic] = text
        last_topic = topic
        session["topic_memory"] = topic_memory
        session["last_topic"]   = last_topic
        return text

    is_short_query  = len(text.split()) < 4
    has_follow_word = any(word in text for word in follow_words)

    if (is_short_query or has_follow_word) and last_topic:
        previous = topic_memory.get(last_topic)
        if previous:
            return previous + " " + text

    return text


# ===============================
# LLM Generation — Ollama (local)
# ===============================
def generate_with_llm(question, context, history):
    history_text = "\n".join(history[-6:])

    MAX_CONTEXT_WORDS = 400
    context = " ".join(context.split()[:MAX_CONTEXT_WORDS])

    prompt = f"""You are an AI assistant for College of Engineering Chengannur. You only answer questions using the information provided below.

STRICT RULES:
- Use ONLY the information provided in the Information section below.
- If the Information section does not contain a clear answer, say: "I don't have that information right now. Please contact the college administration office."
- NEVER invent facts, names, events, or details not explicitly in the Information section.
- NEVER mention document names, PDF files, sources, or use words like "context", "document", "file", or "PDF".
- Do NOT use your general training knowledge to fill in gaps.

Response Style:
- Write 2-4 clear sentences.
- Be helpful and conversational.

Previous conversation:
{history_text}

Information:
{context}

Student question:
{question}

Answer (only using the Information above):"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=60
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

    # Per-session state
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
            "Hello! I'm your AI college assistant for College of Engineering Chengannur. "
            "I can help with exams, fees, faculty information, "
            "library services, campus events, and college activities."
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
            "library services, college events like Arts Festival and Sports Meet, "
            "and student forums like IEEE, IEDC, TinkerHub and GDC."
        )
        _save_history(user_text, response)
        return jsonify({"intent": "help", "response": response})

    # ===============================
    # Query Enrichment
    # ===============================
    enriched_query = enrich_query(user_text)

    # ===============================
    # Intent Detection
    # ===============================
    vec    = vectorizer.transform([user_text])
    probs  = model.predict_proba(vec)[0]
    intent = "unknown" if max(probs) < 0.2 else model.classes_[probs.argmax()]

    # ===============================
    # Document Retrieval + LLM Response
    # ===============================
    doc_name, content, used_fallback = find_most_relevant_document(enriched_query)

    if content:
        print("\n--- Context sent to LLM ---")
        print(content[:500])
        print("--- End of Context ---\n")

        if used_fallback:
            _log_low_confidence(user_text)

        response = generate_with_llm(
            user_text,
            content,
            session["conversation_history"]
        )
    else:
        _log_low_confidence(user_text)
        response = (
            "I couldn't find that information in the available documents. "
            "Please contact the college administration office."
        )

    _save_history(user_text, response)
    return jsonify({"intent": intent, "response": response})


# ===============================
# Helpers
# ===============================
def _save_history(user_text, response):
    history = session.get("conversation_history", [])
    history.append("User: " + user_text)
    history.append("Bot: "  + response)
    if len(history) > 20:
        history = history[-20:]
    session["conversation_history"] = history


def _log_low_confidence(query):
    try:
        with open("logs/low_confidence.txt", "a") as f:
            f.write(query + "\n")
    except Exception:
        pass


# ===============================
# Run Server
# ===============================
if __name__ == "__main__":
    app.run(debug=True)