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
last_user_query = ""
current_topic = None


# ===============================
# Detect Topic
# ===============================
def detect_topic(user_text):

    text = user_text.lower()

    if "faculty" in text or "teacher" in text or "hod" in text:
        return "faculty"

    if "exam" in text:
        return "exam"

    if "fee" in text:
        return "fees"

    if "library" in text:
        return "library"

    if "event" in text:
        return "event"

    return None


# ===============================
# Follow-up Query Enrichment
# ===============================
def enrich_query_with_memory(user_text):

    global last_user_query
    global current_topic

    follow_words = [
        "its", "their", "them", "that",
        "those", "what about", "and",
        "also", "another"
    ]

    text = user_text.lower()

    for word in follow_words:

        if word in text:

            if current_topic:
                return current_topic + " " + user_text

            if last_user_query:
                return last_user_query + " " + user_text

    return user_text


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

        return "Sorry, I couldn't generate a response right now."

    except Exception:
        return "The AI response service is currently unavailable."


# ===============================
# Chat API
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    global conversation_history
    global last_user_query
    global current_topic

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"response": "Invalid request"}), 400

    user_text = data["message"].strip()
    lower_text = user_text.lower()

    # ===============================
    # Greeting
    # ===============================
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]

    if lower_text in greetings:

        response = (
            "Hello! I'm your AI college assistant. "
            "I can help with information about exams, fees, faculty members, "
            "library services, campus facilities, and college events. "
            "What would you like to know?"
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
            "I can assist you with information about exam schedules, "
            "fee structures, faculty details, library services, "
            "campus facilities, administrative contacts, and college events."
        )

        conversation_history.append("User: " + user_text)
        conversation_history.append("Bot: " + response)

        return jsonify({"intent": "help", "response": response})


    # ===============================
    # Topic Detection
    # ===============================
    detected_topic = detect_topic(user_text)

    if detected_topic:
        current_topic = detected_topic


    # ===============================
    # ML Intent Detection
    # ===============================
    vec = vectorizer.transform([user_text])
    probs = model.predict_proba(vec)[0]
    max_prob = max(probs)

    if max_prob < 0.2:
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
            "Students are required to maintain at least 75% attendance "
            "in each subject. If attendance falls below this requirement, "
            "the student may not be eligible to appear for the semester examinations."
        )


    elif intent in [
        "exam",
        "fees",
        "event",
        "faculty",
        "admin",
        "library",
        "syllabus",
        "result"
    ] or any(word in lower_text for word in [
        "exam",
        "fee",
        "library",
        "event",
        "faculty",
        "result",
        "facility"
    ]):

        # ===============================
        # Follow-up Query Enrichment
        # ===============================
        enriched_query = enrich_query_with_memory(user_text)

        expanded_query = enriched_query + " details information explanation"

        doc_name, content = find_most_relevant_document(expanded_query)

        if content:

            print("\n--- Context sent to LLM ---")
            print(content)
            print("--- End of Context ---\n")

            response = generate_with_llm(user_text, content)

        else:

            response = (
                "I couldn't find that information in the available documents. "
                "Please contact the college administration office for clarification."
            )


    else:

        response = (
            "I can assist with college information such as exams, fees, "
            "faculty details, library services, and campus facilities."
        )


    # ===============================
    # Save Conversation Memory
    # ===============================
    conversation_history.append("User: " + user_text)
    conversation_history.append("Bot: " + response)

    last_user_query = user_text

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