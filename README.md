# 📩 SMS Spam Classification & Filtering System

A Machine Learning and Natural Language Processing (NLP) based system that classifies SMS messages as **Spam** or **Safe**.

This project builds a spam detection model using a labeled SMS dataset and deploys it with a simple web interface using **Streamlit**.

---

# 🚀 Project Overview

Spam SMS messages are unsolicited messages that often contain advertisements, phishing links, or scams. These messages can cause privacy issues and security risks.

This project uses **Machine Learning and Natural Language Processing** to automatically detect whether a message is **Spam** or **Safe**.

The system processes the text data, converts it into numerical features, trains a machine learning model, and then predicts the class of new SMS messages entered by the user.

---

# 🧠 Technologies Used

- Python
- Machine Learning
- Natural Language Processing (NLP)
- Scikit-learn
- NLTK
- Pandas
- Streamlit
- TF-IDF Vectorization
- Multinomial Naive Bayes

---

# 📂 Project Structure


SMS-Spam-Classifier
│
├── dataset
│ └── spam.csv
│
├── models
│ ├── spam_model.pkl
│ └── vectorizer.pkl
│
├── notebook
│ └── spam_classifier.ipynb
│
├── app.py
├── web_app.py
├── requirements.txt
└── README.md


---

# ⚙️ How the System Works

1️⃣ **Data Collection**
- SMS dataset containing spam and safe messages.

2️⃣ **Text Preprocessing**
- Convert text to lowercase
- Remove punctuation
- Remove stopwords
- Apply stemming

3️⃣ **Feature Extraction**
- TF-IDF Vectorization converts text into numeric vectors.

4️⃣ **Model Training**
- Multinomial Naive Bayes classifier is trained on the dataset.

5️⃣ **Prediction**
- The trained model predicts whether a message is Spam or Safe.

6️⃣ **Web Interface**
- A Streamlit web app allows users to input SMS messages and see predictions instantly.

---

# 🛠 Installation

### 1️⃣ Clone or Download the Project


git clone https://github.com/yourusername/SMS-Spam-Classifier.git


or download the ZIP and extract it.

---

### 2️⃣ Navigate to the Project Folder


cd SMS-Spam-Classifier


---

### 3️⃣ Install Dependencies


pip install -r requirements.txt


---

# ▶️ Running the Project

### Train the Model


python app.py


This will:
- Load the dataset
- Train the ML model
- Save the trained model

---

### Run the Web Application


streamlit run web_app.py


The web application will open in your browser:


http://localhost:8501


---

# 💡 Example Messages

### Spam Messages


Congratulations! You have won a free lottery ticket. Claim now.

URGENT! You have been selected for a $1000 gift card.


---

### Safe Messages


Hey, are we meeting at 6 pm today?

Please send me the project report.


---

# 📊 Model Used

**Multinomial Naive Bayes**

Why this model?
- Works well for text classification
- Fast training
- High accuracy for spam filtering

---

# 🔮 Future Improvements

- Add deep learning models (LSTM / BERT)
- Improve UI design
- Deploy online using Streamlit Cloud
- Add message history database
- Add SMS API integration

---

# 📚 Learning Outcomes

This project demonstrates:

- Machine Learning pipeline
- Natural Language Processing techniques
- Text feature extraction
- Model training and evaluation
- Web app deployment using Streamlit

---

# 👨‍💻 Author

**Mani Kumar Chapa**

Machine Learning Project  
SMS Spam Classification System

---
