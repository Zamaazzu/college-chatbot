import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents data
with open("data/intents.json", "r") as f:
    intents = json.load(f)

texts = []
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# Save the trained model and vectorizer
pickle.dump(model, open("models/intent_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/tfidf.pkl", "wb"))

print("Intent classification model trained and saved successfully!")
