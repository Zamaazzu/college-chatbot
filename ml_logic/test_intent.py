import pickle

# Load trained model and vectorizer
model = pickle.load(open("models/intent_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))

while True:
    user_input = input("Enter a question (or 'exit' to quit): ")

    if user_input.lower() == "exit":
        break

    vec = vectorizer.transform([user_input])
    intent = model.predict(vec)[0]

    print("Predicted intent:", intent)
