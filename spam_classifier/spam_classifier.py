import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv")

# Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Convert text to numerical form
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Check accuracy
predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))

# Test with a new email
test_email = "Congratulations! You have won free money"
test_email_vec = vectorizer.transform([test_email])
result = model.predict(test_email_vec)

print("Test Email Result:", "Spam" if result[0] == 1 else "Not Spam")

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model saved successfully")