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

# Theme colors
if theme == "🌌 Neon Cyberpunk":
    background = "linear-gradient(-45deg,#0f0c29,#302b63,#ff00cc,#00ffff)"
    text = "#00ffff"
    heading = "#ff00ff"
    card = "rgba(0,0,0,0.6)"

elif theme == "🌙 Dark Mode":
    background = "#0e1117"
    text = "#ffffff"
    heading = "#00ffff"
    card = "#1e1e1e"

elif theme == "☀ Light Mode":
    background = "#ffffff"
    text = "#000000"
    heading = "#ff4b4b"
    card = "#f3f3f3"

else:  # Glass UI
    background = "linear-gradient(to right,#141e30,#243b55)"
    text = "#ffffff"
    heading = "#00ffff"
    card = "rgba(255,255,255,0.1)"

# ===============================
# CSS
# ===============================

st.markdown(f"""
<style>

.stApp {{
    background: {background};
}}

h1,h2,h3 {{
    color: {heading};
}}

p, label {{
    color: {text};
}}

/* Sidebar */

section[data-testid="stSidebar"] {{
    background-color:#0a0a0a;
}}

/* Cards / containers */

.block-container {{
    padding-top:2rem;
}}

div[data-testid="stFileUploader"] {{
    background:{card};
    padding:20px;
    border-radius:10px;
}}

/* Chat input */

div[data-testid="stChatInput"] textarea {{
    background:#111;
    color:#ffffff;
    border:2px solid #ff00ff;
}}

div[data-testid="stChatInput"] textarea::placeholder {{
    color:#aaaaaa;
}}

/* Upload box */

[data-testid="stFileUploaderDropzone"] {{
    background:{card};
    color:{text};
}}

/* Buttons */

div.stButton > button {{
    background:#000;
    color:#00ffff;
    border:2px solid #ff00ff;
    border-radius:8px;
}}

</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================

@st.cache_resource
def load_model():
    model = joblib.load("spam_classifier/spam_model.pkl")
    vectorizer = joblib.load("spam_classifier/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# SIDEBAR STATS
# ===============================

st.sidebar.header("📊 Model Statistics")

data = pd.read_csv("spam_classifier/spam.csv")
data['label'] = data['label'].map({'ham':0,'spam':1})

X = vectorizer.transform(data['text'])
y = data['label']

y_pred = model.predict(X)

acc = accuracy_score(y,y_pred)

st.sidebar.write(f"Accuracy: {round(acc*100,2)}%")

# Accuracy chart
fig,ax = plt.subplots()
ax.bar(["Accuracy"],[acc*100])
ax.set_ylim([0,100])
st.sidebar.pyplot(fig)

# Confusion matrix
cm = confusion_matrix(y,y_pred)

fig2,ax2 = plt.subplots()

ax2.imshow(cm)

ax2.set_xticks([0,1])
ax2.set_yticks([0,1])

ax2.set_xticklabels(["Ham","Spam"])
ax2.set_yticklabels(["Ham","Spam"])

for i in range(2):
    for j in range(2):
        ax2.text(j,i,cm[i,j],ha="center")

st.sidebar.pyplot(fig2)

# ===============================
# MAIN UI
# ===============================

st.title("🤖 Spam Mail Detector AI")

if "messages" not in st.session_state:
    st.session_state.messages=[]

user_input = st.chat_input("Type your email message...")

if user_input:

    st.session_state.messages.append(("user",user_input))

    vec = vectorizer.transform([user_input])

    pred = model.predict(vec)
    prob = model.predict_proba(vec)[0]

    spam_prob = round(prob[1]*100,2)

    if pred[0]==1:
        reply=f"🚨 SPAM DETECTED ({spam_prob}% confidence)"
    else:
        reply=f"✅ SAFE MESSAGE ({100-spam_prob}% confidence)"

    st.session_state.messages.append(("assistant",reply))

for role,msg in st.session_state.messages:

    st.chat_message(role).write(msg)

st.divider()

# ===============================
# FILE UPLOAD
# ===============================

st.header("📩 Upload Email File (.txt)")

uploaded = st.file_uploader("Upload a text file",type=["txt"])

if uploaded:

    content = uploaded.read().decode("utf-8")

    st.write("File Content:")
    st.write(content)

    vec = vectorizer.transform([content])

    pred = model.predict(vec)

    if pred[0]==1:

        st.error("❌ This file is SPAM")

    else:

        st.success("✅ This file is NOT SPAM")
