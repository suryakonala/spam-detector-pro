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
# THEME SWITCHER
# ===============================

theme = st.sidebar.radio(
    "🌈 Select Theme",
    ["🌌 Neon Cyberpunk", "🌙 Dark Mode", "☀ Light Mode", "💎 Glass UI"]
)

if theme == "🌌 Neon Cyberpunk":
    bg = "linear-gradient(-45deg,#0f0c29,#302b63,#ff00cc,#00ffff)"
    text = "#ffffff"
    heading = "#ff00ff"
    card = "rgba(0,0,0,0.6)"

elif theme == "🌙 Dark Mode":
    bg = "#0e1117"
    text = "#ffffff"
    heading = "#00ffff"
    card = "#1c1c1c"

elif theme == "☀ Light Mode":
    bg = "#ffffff"
    text = "#000000"
    heading = "#ff4b4b"
    card = "#f5f5f5"

else:
    bg = "linear-gradient(to right,#141e30,#243b55)"
    text = "#ffffff"
    heading = "#00ffff"
    card = "rgba(255,255,255,0.1)"

# ===============================
# GLOBAL CSS FIX
# ===============================

st.markdown(f"""
<style>

.stApp {{
background:{bg};
}}

h1,h2,h3 {{
color:{heading};
}}

p, label, span {{
color:{text};
}}

.block-container {{
padding-top:2rem;
}}

section[data-testid="stSidebar"] {{
background:#000;
}}

/* CHAT INPUT */

div[data-testid="stChatInput"] textarea {{
background:#000 !important;
color:#ffffff !important;
border:2px solid #ff00ff !important;
}}

div[data-testid="stChatInput"] textarea::placeholder {{
color:#bbbbbb !important;
}}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {{
background:{card} !important;
padding:20px;
border-radius:10px;
}}

[data-testid="stFileUploaderDropzone"] {{
background:#1e1e1e !important;
color:#ffffff !important;
}}

[data-testid="stFileUploaderDropzone"] span {{
color:#ffffff !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"] {{
color:#ffffff !important;
}}

/* UPLOAD BUTTON */

button[kind="secondary"] {{
background:#ffffff !important;
color:#000000 !important;
border-radius:6px;
}}

/* NORMAL BUTTONS */

div.stButton > button {{
background:#000;
color:#00ffff;
border:2px solid #ff00ff;
}}

/* TEXT AREA */

textarea {{
background:#111 !important;
color:#ffffff !important;
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
# SIDEBAR MODEL STATS
# ===============================

st.sidebar.header("📊 Model Statistics")

data = pd.read_csv("spam_classifier/spam.csv")
data['label'] = data['label'].map({'ham':0,'spam':1})

X = vectorizer.transform(data['text'])
y = data['label']

pred = model.predict(X)

acc = accuracy_score(y,pred)

st.sidebar.write(f"Accuracy: {round(acc*100,2)}%")

fig,ax = plt.subplots()
ax.bar(["Accuracy"],[acc*100])
ax.set_ylim([0,100])
st.sidebar.pyplot(fig)

cm = confusion_matrix(y,pred)

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
