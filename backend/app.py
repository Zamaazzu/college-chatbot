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
# Conversation Memory (temporary)
# ===============================
conversation_history = []

# ===============================
# LLM Generation Function
# ===============================
def generate_with_llm(question, context):
    global conversation_history

    # Take last 6 conversation messages
    history_text = "\n".join(conversation_history[-6:])

    prompt = f"""
You are a professional college assistant chatbot.

Follow these STRICT rules:
1. Use ONLY the provided context.
2. Do NOT guess.
3. Do NOT calculate.
4. If answer is not present in context, say:
"I do not have that information in the official documents."
5. Be concise and professional.

Previous conversation:
{history_text}

Context:
{context}

User Question:
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

        return response.json()["response"]

    except Exception as e:
        return "LLM service is currently unavailable."

# ===============================
# API Route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    global conversation_history

    data = request.get_json()
    user_text = data["message"].strip()

    # ===============================
    # Greeting Shortcut
    # ===============================
    if user_text.lower() in ["hi", "hello", "hey"]:
        greeting_response = "Hello! I’m your college assistant. How can I help you?"

        conversation_history.append("User: " + user_text)
        conversation_history.append("Bot: " + greeting_response)

        return jsonify({
            "intent": "greeting",
            "response": greeting_response
        })

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
    # Response Logic
    # ===============================
    if intent == "attendance":
        response = (
            "The minimum attendance required is 75%. "
            "Personal attendance details will be available "
            "after integration with the college attendance system."
        )

    elif intent in ["exam", "fees", "event","faculity","admin", "library", "syllabus", "result"]:
        doc_name, content = find_most_relevant_document(user_text)

        if content:
            print("\n--- Context sent to LLM ---")
            print(content)
            print("--- End of Context ---\n")

            response = generate_with_llm(user_text, content)
        else:
            response = "No relevant document found in official records."

    else:
        response = "Sorry, I can currently help only with college-related queries."

    # ===============================
    # Save Conversation Memory
    # ===============================
    conversation_history.append("User: " + user_text)
    conversation_history.append("Bot: " + response)

    # Limit memory size (avoid infinite growth)
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