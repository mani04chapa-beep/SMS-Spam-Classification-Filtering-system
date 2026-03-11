import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

# Load trained model
model = pickle.load(open("models/spam_model.pkl","rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl","rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-z0-9\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [ps.stem(word) for word in words]

    return " ".join(words)


# Streamlit page settings
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Title
st.title("📩 SMS Spam Detection System")
st.write("Detect whether a message is **Spam or Safe** using Machine Learning.")

# Example messages
with st.expander("💡 Try Example Messages"):
    st.write("Spam Example:")
    st.code("Congratulations! You won a free lottery ticket")

    st.write("Safe Example:")
    st.code("Hey, are we meeting at 6 pm today?")

# Input box
sms = st.text_area("Enter SMS Message", height=150)

# Prediction
if st.button("🔍 Predict"):

    if sms.strip() == "":
        st.warning("Please enter a message first.")
    else:

        cleaned = clean_text(sms)

        vector = vectorizer.transform([cleaned])

        result = model.predict(vector)[0]

        prob = model.predict_proba(vector)[0]

        spam_prob = prob[1]
        safe_prob = prob[0]

        if result == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **SAFE**")

        st.subheader("Prediction Confidence")

        st.write(f"Safe Message Probability: **{safe_prob:.2f}**")
        st.progress(float(safe_prob))

        st.write(f"Spam Probability: **{spam_prob:.2f}**")
        st.progress(float(spam_prob))


# Prediction history
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Save to History"):

    if sms.strip() != "":
        st.session_state.history.append(sms)

# Show history
if st.session_state.history:

    st.subheader("🕘 Previous Messages")

    for msg in st.session_state.history[-5:]:
        st.write("•", msg)

# Footer
st.markdown("---")
st.write("Built with Machine Learning + Streamlit")