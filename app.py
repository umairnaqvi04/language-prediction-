import streamlit as st
import joblib
import os
import re

# ================= UI THEME =================
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}
.stTextArea textarea {
    background-color: #1e293b;
    color: #e2e8f0;
}
.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}
.stButton button:hover {
    background-color: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD =================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "language_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# ================= CLEAN =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# ================= UI =================
st.title("🌐 Language Detection System")

text = st.text_area("Enter Text")

if st.button("Detect Language 🚀"):

    if not text.strip():
        st.error("⚠ Please enter text")
    elif len(text.split()) < 2:
        st.warning("⚠ Enter at least 2-3 words")
    else:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        lang = encoder.inverse_transform([pred])[0]

        st.success(f"👉 Predicted Language: {lang}")

# ================= SIDEBAR =================
st.sidebar.title("📌 Info")
st.sidebar.write("ML Language Detection Project")
st.sidebar.write("Model: SVM / Logistic Regression")
