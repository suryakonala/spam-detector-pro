import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Mail Detector AI",
    page_icon="📧",
    layout="wide"
)

# ---------------- CYBERPUNK THEME ----------------
st.markdown("""
<style>

/* Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #ff00cc, #00ffe7);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
    color: white;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Sidebar Dark */
section[data-testid="stSidebar"] {
    background-color: #0a0a0a;
    color: white;
}

/* Glass Effect Card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,255,231,0.4);
}

/* Title */
.main-title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    color: #00ffe7;
    text-shadow: 0 0 15px #00ffe7;
}

/* Input Label */
.sub-title {
    font-size: 22px;
    font-weight: 600;
    color: white;
}

/* Text Area Fix */
textarea {
    background-color: rgba(0,0,0,0.6) !important;
    color: white !important;
    border: 2px solid #00ffe7 !important;
}

/* Button */
.stButton>button {
    background-color: #00ffe7;
    color: black;
    font-weight: bold;
    border-radius: 8px;
}

/* Result Boxes */
.safe-box {
    background-color: #00c853;
    padding: 18px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}

.spam-box {
    background-color: #d50000;
    padding: 18px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("spam_classifier/spam_model.pkl")
    vectorizer = joblib.load("spam_classifier/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 📊 Model Statistics")
data = pd.read_csv("spam_classifier/spam.csv")
accuracy = 92.31
st.sidebar.write(f"Accuracy: {accuracy}%")
st.sidebar.progress(int(accuracy))

# ---------------- MAIN UI ----------------
st.markdown('<div class="main-title">📧 Spam Mail Detector AI</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="glass">', unsafe_allow_html=True)

st.markdown('<div class="sub-title">✉️ Enter your email below:</div>', unsafe_allow_html=True)

email_text = st.text_area("", height=200)

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
                f'<div class="spam-box">🚨 SPAM DETECTED ({spam_prob:.2f}% confidence)</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="safe-box">✅ SAFE EMAIL ({ham_prob:.2f}% confidence)</div>',
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)
