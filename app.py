import pandas as pd
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords only (no tokenizer needed)
nltk.download('stopwords', quiet=True)

# Load dataset
data = pd.read_csv("dataset/spam.csv", header=None)

# Split into label and message
data[['label','message']] = data[0].str.split('\t', expand=True)

data = data[['label','message']]

print("Dataset Loaded Successfully")
print(data.head())

# Initialize tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):

    if pd.isna(text):
        return ""

    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [ps.stem(word) for word in words]

    return " ".join(words)

# Apply preprocessing
data["clean_message"] = data["message"].apply(clean_text)

# Convert labels
data["label_num"] = data["label"].map({"ham":0, "spam":1})

# Feature extraction
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["clean_message"])
y = data["label_num"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Save model
pickle.dump(model, open("models/spam_model.pkl","wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl","wb"))

print("Model saved successfully!")

# Prediction function
def predict_sms(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    result = model.predict(vector)

    if result[0] == 1:
        return "Spam"
    else:
        return "Not spam"

# Example test
test_message = "Congratulations! You won a free lottery ticket"
print("\nTest Message:", test_message)
print("Prediction:", predict_sms(test_message))

# User input loop
while True:

    sms = input("\nEnter SMS message (type 'exit' to stop): ")

    if sms.lower() == "exit":
        print("Program stopped.")
        break

    print("Prediction:", predict_sms(sms))