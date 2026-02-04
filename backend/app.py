from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load trained ML model
model = pickle.load(open("models/intent_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = data["message"]

    # Simple greeting shortcut (rule-based)
    if user_text.lower().strip() in ["hi", "hello", "hey"]:
        return jsonify({
            "intent": "greeting",
            "response": "Hello! I’m your college assistant. How can I help you?"
        })

    # ML-based intent detection
    vec = vectorizer.transform([user_text])
    probs = model.predict_proba(vec)[0]
    max_prob = max(probs)

    if max_prob < 0.3:
        intent = "unknown"
    else:
        intent = model.classes_[probs.argmax()]

    # Response logic
    if intent == "attendance":
        response = (
            "This query is related to attendance policy. "
            "The minimum attendance required is 75%. "
            "Personal attendance details will be available "
            "after integration with the college attendance system."
        )

    elif intent == "greeting":
        response = "Hello! I’m your college assistant. How can I help you?"

    elif intent in ["exam", "fees", "event", "admin", "library", "syllabus", "result"]:
        response = (
            "This information has not been updated yet. "
            "It will be available once official college data is integrated."
        )

    else:
        response = (
            "Sorry, I can currently help only with college-related queries."
        )

    return jsonify({
        "intent": intent,
        "response": response
    })

if __name__ == "__main__":
    app.run(debug=True)
