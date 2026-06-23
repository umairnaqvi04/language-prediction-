import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Language Detection System",
    page_icon="🌐",
    layout="wide"
)

# =====================
# LOAD MODELS
# =====================
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

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
# SIDEBAR MENU
# =====================
menu = st.sidebar.radio(
    "🌐 Navigation",
    ["🏠 Home", "📊 Dataset Info", "📈 Model Performance", "🔮 Prediction", "ℹ️ About"]
)

# =====================
# HOME PAGE
# =====================
if menu == "🏠 Home":
    st.title("🌐 Language Detection System")
    st.write("Machine Learning based language detection app")

    col1, col2, col3 = st.columns(3)

    col1.metric("Languages", "17")
    col2.metric("Dataset Size", "10,000+")
    col3.metric("Best Model", "Linear SVM")

    st.success("Go to Prediction page from sidebar")

# =====================
# DATASET PAGE
# =====================
elif menu == "📊 Dataset Info":
    st.title("📊 Dataset Overview")

    try:
        df = pd.read_csv("dataset.csv")
        st.dataframe(df.head(10))

        st.bar_chart(df["language"].value_counts())

    except:
        st.warning("dataset.csv file not found")

# =====================
# MODEL PAGE
# =====================
elif menu == "📈 Model Performance":
    st.title("📈 Model Accuracy")

    data = {
        "Model": ["Linear SVM", "KNN", "Naive Bayes"],
        "Accuracy": [98.7, 95.2, 93.5]
    }

    df = pd.DataFrame(data)
    st.dataframe(df)

    st.bar_chart(df.set_index("Model"))

# =====================
# PREDICTION PAGE
# =====================
elif menu == "🔮 Prediction":
    st.title("🔮 Live Language Prediction")

    text = st.text_area("Enter Text")

    if st.button("Detect Language"):

        if not text.strip():
            st.warning("Please enter text")

        elif len(text.split()) < 2:
            st.warning("Enter at least 2-3 words")

        else:
            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            language = encoder.inverse_transform([pred])[0]

            st.success(f"✅ Predicted Language: {language}")

# =====================
# ABOUT PAGE
# =====================
elif menu == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.write("""
    - Machine Learning Project
    - Language Detection using NLP
    - Models: SVM, KNN, Naive Bayes
    """)

    st.info("Developed by: Umair Naqvi")
