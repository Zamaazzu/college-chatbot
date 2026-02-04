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

    # Vectorize input
    vec = vectorizer.transform([user_text])

    # Get prediction probabilities
    probs = model.predict_proba(vec)[0]
    max_prob = max(probs)

    # Confidence threshold check
    if max_prob < 0.25:
        intent = "unknown"
    else:
        intent = model.classes_[probs.argmax()]

    return jsonify({
        "intent": intent
    })


if __name__ == "__main__":
    app.run(debug=True)
