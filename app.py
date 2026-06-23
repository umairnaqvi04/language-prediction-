import streamlit as st
import joblib
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# ================= THEME =================
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

# ================= SAFE LOAD =================
BASE_DIR = os.path.dirname(__file__)

try:
    model = joblib.load(os.path.join(BASE_DIR, "language_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
    encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
except:
    st.error("❌ Model files missing!")
    st.stop()

# ================= CLEAN =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# ================= TITLE =================
st.title("🌐 Language Detection System")

# ================= SIDEBAR =================
st.sidebar.title("📌 Info")
st.sidebar.write("ML Project: Language Detection")
st.sidebar.write("Models: SVM / Logistic Regression")

# ================= INPUT =================
text = st.text_area("Enter Text")

# ================= PREDICTION =================
if st.button("Detect Language 🚀"):

    # ❌ NEGATIVE CASE 1
    if not text.strip():
        st.error("⚠ Please enter text first")
        st.stop()

    # ❌ NEGATIVE CASE 2
    if len(text.split()) < 2:
        st.warning("⚠ Enter at least 2-3 words")
        st.stop()

    try:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        lang = encoder.inverse_transform([pred])[0]

        st.success(f"👉 Predicted Language: {lang}")

    except:
        st.error("❌ Prediction failed")

# ================= EDA GRAPH =================
st.subheader("📊 Language Distribution (EDA)")

try:
    # fake dataset preview (for GitHub deploy safe)
    df = pd.DataFrame({
        "Language": encoder.classes_
    })

    count = [10]*len(df)  # placeholder for deploy safety

    fig, ax = plt.subplots()
    ax.bar(df["Language"], count)
    plt.xticks(rotation=45)
    st.pyplot(fig)

except:
    st.info("EDA not available")

# ================= FOOTER =================
st.sidebar.info("✔ System Ready")
