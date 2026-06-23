import streamlit as st
import joblib
import os
import re

st.set_page_config(page_title="Language Detection", page_icon="🌐")

st.title("🌐 Language Detection System")

# =====================
# CHECK FILES FIRST
# =====================
required_files = [
    "language_model.pkl",
    "vectorizer.pkl",
    "label_encoder.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"❌ Missing file: {file}")
        st.stop()

# =====================
# LOAD MODELS SAFELY
# =====================
@st.cache_resource
def load_models():
    model = joblib.load("language_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, encoder

model, vectorizer, encoder = load_models()

# =====================
# CLEAN TEXT
# =====================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =====================
# UI
# =====================
text = st.text_area("Enter Text")

if st.button("Detect Language"):

    if not text.strip():
        st.warning("Enter text first")

    else:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        language = encoder.inverse_transform([pred])[0]

        st.success(f"✅ Predicted Language: {language}")
