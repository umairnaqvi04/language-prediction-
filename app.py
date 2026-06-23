import streamlit as st
import joblib
import re

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Language Detection System",
    page_icon="🌐",
    layout="centered"
)

# ==========================
# LOAD MODELS
# ==========================
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

# ==========================
# CLEAN TEXT FUNCTION
# ==========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==========================
# TITLE
# ==========================
st.title("🌐 Language Detection System")
st.write("Enter any text and detect its language using Machine Learning.")

# ==========================
# INPUT BOX
# ==========================
text = st.text_area("Enter Text", height=150)

# ==========================
# PREDICT BUTTON
# ==========================
if st.button("Detect Language 🚀"):

    if text.strip() == "":
        st.warning("Please enter some text")

    else:
        cleaned = clean_text(text)

        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]

        language = encoder.inverse_transform([pred])[0]

        st.success(f"👉 Predicted Language: **{language}**")

# ==========================
# SIDEBAR INFO
# ==========================
st.sidebar.title("ℹ️ About")

st.sidebar.write("""
This is a Machine Learning project for language detection.

- Model: Logistic Regression / SVM  
- Vectorizer: TF-IDF (Character n-grams)  
- Accuracy: High precision text classification
""")
