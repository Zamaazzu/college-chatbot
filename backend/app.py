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
            "Minimum attendance required is 75%. "
            "Students falling below this may require condonation "
            "as per college rules."
        )
    elif intent in ["exam", "fees", "event", "admin"]:
        response = (
            "This information has not been updated yet. "
            "It will be available once official college data is integrated."
        )
    else:
        response = (
            "Sorry, I can currently help only with college-related queries "
            "such as attendance, exams, fees, and events."
        )

    return jsonify({
        "intent": intent,
        "response": response
    })


if __name__ == "__main__":
    app.run(debug=True)
