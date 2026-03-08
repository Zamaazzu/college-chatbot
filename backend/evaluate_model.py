import json
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ====================================
# Locate intents.json inside data folder
# ====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # backend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                   # college-chatbot/
file_path = os.path.join(PROJECT_ROOT, "data", "intents.json")

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)


# ====================================
# Prepare dataset
# ====================================
texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])


# ====================================
# Train/Test Split
# ====================================
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)


# ====================================
# Vectorization
# ====================================
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ====================================
# Train Model
# ====================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


# ====================================
# Evaluate
# ====================================
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Intent Model Evaluation")
print("==============================")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))