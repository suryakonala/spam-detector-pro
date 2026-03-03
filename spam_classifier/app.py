import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

st.set_page_config(
    page_title="Spam Mail Detector AI",
    page_icon="📧",
    layout="wide"
)

# ===============================
# 🌌 CYBERPUNK THEME (FINAL)
# ===============================

st.markdown("""
<style>

/* Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #ff00cc, #00ffff);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* General Text */
body, p, div, span, label {
    color: #00ffff !important;
}

/* Headings */
h1, h2, h3 {
    color: #ff00ff !important;
    text-shadow: 0 0 20px #ff00ff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #00ffff !important;
}

/* Chat Input Box */
div[data-testid="stChatInput"] textarea {
    background-color: #000000 !important;
    color: #ffffff !important;   /* typed text visible */
    border: 2px solid #ff00ff !important;
    font-size: 18px !important;
}

/* Placeholder */
div[data-testid="stChatInput"] textarea::placeholder {
    color: #00ffff !important;
    opacity: 1 !important;
}

/* Send Button */
div[data-testid="stChatInput"] button {
    background-color: #111 !important;
    color: #00ffff !important;
    border: 2px solid #ff00ff !important;
}

/* Buttons */
div.stButton > button {
    background: black !important;
    color: #00ffff !important;
    border: 2px solid #ff00ff !important;
    box-shadow: 0 0 15px #ff00ff;
    font-weight: bold;
}

div.stButton > button:hover {
    box-shadow: 0 0 25px #00ffff;
}

/* Text Area */
textarea {
    background-color: #111 !important;
    color: #00ffff !important;
    border: 2px solid #ff00ff !important;
}

/* Remove white container */
.block-container {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL (DEPLOYMENT SAFE)
# ===============================

@st.cache_resource
def load_model():
    model = joblib.load("spam_classifier/spam_model.pkl")
    vectorizer = joblib.load("spam_classifier/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# SIDEBAR MODEL STATISTICS
# ===============================

st.sidebar.header("📊 Model Statistics")

data = pd.read_csv("spam_classifier/spam.csv")
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = vectorizer.transform(data['text'])
y = data['label']
y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
st.sidebar.write(f"Accuracy: {round(accuracy*100,2)}%")

# Accuracy Chart
fig1, ax1 = plt.subplots()
ax1.bar(["Accuracy"], [accuracy*100], color="#00ffff")
ax1.set_ylim([0,100])
ax1.set_facecolor("#111")
fig1.patch.set_facecolor("#111")
ax1.tick_params(colors='white')
st.sidebar.pyplot(fig1)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

fig2, ax2 = plt.subplots()
ax2.imshow(cm, cmap="magma")
ax2.set_xticks([0,1])
ax2.set_yticks([0,1])
ax2.set_xticklabels(["Ham","Spam"])
ax2.set_yticklabels(["Ham","Spam"])
ax2.set_facecolor("#111")
fig2.patch.set_facecolor("#111")
ax2.tick_params(colors='white')

for i in range(2):
    for j in range(2):
        ax2.text(j, i, cm[i,j], ha="center", va="center", color="white")

st.sidebar.pyplot(fig2)

# ===============================
# MAIN CHAT SECTION
# ===============================

st.markdown("## 🤖 Spam Mail Detector AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    email_vector = vectorizer.transform([user_input])
    prediction = model.predict(email_vector)
    prob = model.predict_proba(email_vector)[0]
    spam_probability = round(prob[1] * 100, 2)

    if prediction[0] == 1:
        reply = f"🚨 SPAM DETECTED ({spam_probability}% confidence)"
    else:
        reply = f"✅ MESSAGE SAFE ({100 - spam_probability}% confidence)"

    st.session_state.messages.append(("assistant", reply))

for role, message in st.session_state.messages:
    st.chat_message(role).write(message)

st.markdown("---")

# ===============================
# FILE UPLOAD SECTION
# ===============================

st.markdown("## 📩 Upload Email File (.txt)")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    st.write("File Content:")
    st.write(content)

    email_vector = vectorizer.transform([content])
    prediction = model.predict(email_vector)

    if prediction[0] == 1:
        st.markdown("<h3 style='color:#ff4b4b;'>❌ This file is SPAM</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:#00ff99;'>✅ This file is NOT SPAM</h3>", unsafe_allow_html=True)
