from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_logic.document_retriever import find_most_relevant_document
import pickle
import requests

app = Flask(__name__)
CORS(app)

# Load trained ML model
model = pickle.load(open("models/intent_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))


# 🔹 LLM generation function (NOT a route)
def generate_with_llm(question, context):
    prompt = f"""
    You are a professional college assistant chatbot.

    Your task:
    1. Read the provided context carefully.
    2. Extract the exact answer to the question.
    3. Do NOT guess.
    4. Do NOT calculate.
    5. If multiple values exist, choose the one that matches the question precisely.
    6. If answer is not present, say:
    "I do not have that information in the official documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """



    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


# 🔹 This is your API route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = data["message"]

    # Greeting shortcut
    if user_text.lower().strip() in ["hi", "hello", "hey"]:
        return jsonify({
            "intent": "greeting",
            "response": "Hello! I’m your college assistant. How can I help you?"
        })

    # ML intent detection
    vec = vectorizer.transform([user_text])
    probs = model.predict_proba(vec)[0]
    max_prob = max(probs)

    if max_prob < 0.2:
        intent = "unknown"
    else:
        intent = model.classes_[probs.argmax()]

    # 🔹 Response logic
    if intent == "attendance":
        response = (
            "The minimum attendance required is 75%. "
            "Personal attendance details will be available "
            "after integration with the college attendance system."
        )

    elif intent in ["exam", "fees", "event", "admin", "library", "syllabus", "result"]:
        doc_name, content = find_most_relevant_document(user_text)

        if content:
            #  Now we use LLM here
            print("\n--- Context sent to LLM ---")
            print(content)
            print("--- End of Context ---\n")
            response = generate_with_llm(user_text, content)
        else:
            response = "No relevant document found in official records."

    else:
        response = "Sorry, I can currently help only with college-related queries."

    return jsonify({
        "intent": intent,
        "response": response
    })


if __name__ == "__main__":
    app.run(debug=True)
