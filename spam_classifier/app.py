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
# 🎨 THEME SWITCHER
# ===============================

theme = st.sidebar.radio(
    "🌈 Select Theme",
    ["🌌 Neon Cyberpunk", "🌙 Dark Mode", "☀ Light Mode", "💎 Glass UI"]
)

if theme == "🌌 Neon Cyberpunk":
    background = "linear-gradient(-45deg, #0f0c29, #302b63, #ff00cc, #00ffff)"
    text_color = "#00ffff"
    heading_color = "#ff00ff"

elif theme == "🌙 Dark Mode":
    background = "#0e1117"
    text_color = "#ffffff"
    heading_color = "#00ffff"

elif theme == "☀ Light Mode":
    background = "#ffffff"
    text_color = "#000000"
    heading_color = "#ff4b4b"

else:  # Glass UI
    background = "linear-gradient(to right, #141e30, #243b55)"
    text_color = "#ffffff"
    heading_color = "#00ffff"

# ===============================
# APPLY THEME CSS
# ===============================

st.markdown(f"""
<style>
.stApp {{
    background: {background};
    background-size: 400% 400%;
}}

body, p, div, span, label {{
    color: {text_color} !important;
}}

h1, h2, h3 {{
    color: {heading_color} !important;
}}

section[data-testid="stSidebar"] {{
    background-color: #0a0a0a !important;
}}

.block-container {{
    background: transparent !important;
}}

div[data-testid="stChatInput"] textarea {{
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 2px solid #ff00ff !important;
    font-size: 18px !important;
}}

div[data-testid="stChatInput"] textarea::placeholder {{
    color: #00ffff !important;
    opacity: 1 !important;
}}

div[data-testid="stChatInput"] button {{
    background-color: #111 !important;
    color: #00ffff !important;
    border: 2px solid #ff00ff !important;
}}

div.stButton > button {{
    background: black !important;
    color: #00ffff !important;
    border: 2px solid #ff00ff !important;
    font-weight: bold;
}}

textarea {{
    background-color: #111 !important;
    color: #00ffff !important;
    border: 2px solid #ff00ff !important;
}}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL (DEPLOY SAFE)
# ===============================

@st.cache_resource
def load_model():
    model = joblib.load("spam_classifier/spam_model.pkl")
    vectorizer = joblib.load("spam_classifier/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# SIDEBAR MODEL STATS
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
ax1.bar(["Accuracy"], [accuracy*100])
ax1.set_ylim([0,100])
st.sidebar.pyplot(fig1)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

fig2, ax2 = plt.subplots()
ax2.imshow(cm)
ax2.set_xticks([0,1])
ax2.set_yticks([0,1])
ax2.set_xticklabels(["Ham","Spam"])
ax2.set_yticklabels(["Ham","Spam"])

for i in range(2):
    for j in range(2):
        ax2.text(j, i, cm[i,j], ha="center", va="center")

st.sidebar.pyplot(fig2)

# ===============================
# MAIN APP
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
# FILE UPLOAD
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
        st.markdown("<h3 style='color:red;'>❌ This file is SPAM</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:green;'>✅ This file is NOT SPAM</h3>", unsafe_allow_html=True)
