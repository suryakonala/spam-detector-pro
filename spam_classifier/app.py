import streamlit as st
import joblib
import pandas as pd
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Spam Mail Detector AI",
    page_icon="📧",
    layout="wide"
)

# ----------------------------
# Custom CSS (Neon Gradient Theme)
# ----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f0036, #ff00cc);
    color: white;
}

.big-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #00ffe7;
}

.sub-text {
    font-size: 20px;
    font-weight: 600;
    color: white;
}

.result-safe {
    background-color: #00c853;
    padding: 15px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}

.result-spam {
    background-color: #d50000;
    padding: 15px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model (Correct Paths)
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("spam_classifier/spam_model.pkl")
    vectorizer = joblib.load("spam_classifier/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# Sidebar - Model Stats
# ----------------------------
st.sidebar.title("📊 Model Statistics")

data = pd.read_csv("spam_classifier/spam.csv")
accuracy = 92.31  # Your trained accuracy (change if needed)

st.sidebar.write(f"Accuracy: {accuracy}%")

st.sidebar.progress(int(accuracy))

# ----------------------------
# Main Title
# ----------------------------
st.markdown('<p class="big-title">📧 Spam Mail Detector AI</p>', unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Input Area
# ----------------------------
st.markdown('<p class="sub-text">✉️ Enter your email message below:</p>', unsafe_allow_html=True)

email_text = st.text_area("Enter Email", height=180)

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("🔍 Analyze Email"):

    if email_text.strip() == "":
        st.warning("Please enter an email message.")
    else:
        transformed = vectorizer.transform([email_text])
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0]

        spam_prob = probability[1] * 100
        ham_prob = probability[0] * 100

        if prediction == 1:
            st.markdown(
                f'<div class="result-spam">🚨 SPAM DETECTED ({spam_prob:.2f}% confidence)</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-safe">✅ SAFE EMAIL ({ham_prob:.2f}% confidence)</div>',
                unsafe_allow_html=True
            )
