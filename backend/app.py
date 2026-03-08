from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_logic.document_retriever import find_most_relevant_document
import pickle
import requests

app = Flask(__name__)
CORS(app)

# ===============================
# Load ML model
# ===============================
model = pickle.load(open("models/intent_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))

# ===============================
# Conversation Memory
# ===============================
conversation_history = []
topic_memory = {}
last_topic = None


# ===============================
# Topic Detection
# ===============================
def detect_topic(text):

    text = text.lower()

    topic_keywords = {
        "faculty": ["faculty", "teacher", "hod", "professor"],
        "exam": ["exam", "semester", "internal", "test"],
        "fees": ["fee", "tuition", "deposit"],
        "library": ["library", "reading", "books"],
        "event": ["event", "fest", "celebration", "sports", "arts"],
        "class": ["class", "lecture", "schedule", "timing"],
        "campus": ["campus", "location", "building", "room"]
    }

    for topic, words in topic_keywords.items():
        for word in words:
            if word in text:
                return topic

    return None


# ===============================
# Follow-up Query Enrichment
# ===============================
def enrich_query(user_text):

    global topic_memory
    global last_topic

    text = user_text.lower()

    follow_words = [
        "its",
        "their",
        "them",
        "that",
        "those",
        "what about",
        "and",
        "also",
        "another",
        "friday",
        "then",
        "next"
    ]

    # detect topic
    topic = detect_topic(text)

    if topic:
        topic_memory[topic] = text
        last_topic = topic
        return text

    # follow-up detection
    for word in follow_words:

        if word in text and last_topic:

            previous_topic_query = topic_memory.get(last_topic)

            if previous_topic_query:
                return previous_topic_query + " " + text

    return text


# ===============================
# LLM Generation
# ===============================
def generate_with_llm(question, context):

    history_text = "\n".join(conversation_history[-6:])

    prompt = f"""
You are an intelligent AI college assistant that helps students with institutional information.

STRICT RULES:
- Use ONLY the information in the context.
- NEVER mention document names, PDF files, sources, or where the information came from.
- NEVER say words like "context", "document", "file", or "PDF".
- Answer naturally as if you already know the information.

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
            }
        )

        result = response.json()

        if "response" in result:
            return result["response"].strip()

        return "Sorry, I couldn't generate a response."

    except Exception:
        return "The AI response service is currently unavailable."


# ===============================
# Chat API
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    global conversation_history

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"response": "Invalid request"}), 400

    user_text = data["message"].strip()
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

        conversation_history.append("User: " + user_text)
        conversation_history.append("Bot: " + response)

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

        conversation_history.append("User: " + user_text)
        conversation_history.append("Bot: " + response)

        return jsonify({"intent": "help", "response": response})

    # ===============================
    # Query Enrichment (Memory)
    # ===============================
    enriched_query = enrich_query(user_text)

    # ===============================
    # Intent Detection
    # ===============================
    vec = vectorizer.transform([user_text])
    probs = model.predict_proba(vec)[0]

    if max(probs) < 0.2:
        intent = "unknown"
    else:
        intent = model.classes_[probs.argmax()]

    # ===============================
    # Attendance Shortcut
    # ===============================
    if "attendance" in lower_text:
        intent = "attendance"

    # ===============================
    # Response Logic
    # ===============================
    if intent == "attendance":

        response = (
            "Students must maintain at least 75 percent attendance in each subject. "
            "Students below this limit may not be eligible to appear for semester examinations."
        )

    else:

        expanded_query = enriched_query + " college information explanation"

        doc_name, content = find_most_relevant_document(expanded_query)

        if content:

            print("\n--- Context sent to LLM ---")
            print(content)
            print("--- End of Context ---\n")

            response = generate_with_llm(user_text, content)

        else:

            response = (
                "I couldn't find that information in the available documents. "
                "Please contact the college administration office."
            )

    # ===============================
    # Save Conversation
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
# Run Server
# ===============================
if __name__ == "__main__":
    app.run(debug=True)