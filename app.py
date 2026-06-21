import streamlit as st
import joblib
import re

# Page Configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)

# Load Model and Vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Text Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Main Title
st.title("📰 AI Based Fake News Detection System")

st.markdown("""
### Detect whether a news article is Real or Fake

This application analyzes news content using Machine Learning and NLP techniques.

Paste a news article or headline below and click **Analyze News**.
""")

# Input Section
st.subheader("📝 News Input")

text = st.text_area(
    "Paste News Article Here",
    height=250,
    placeholder="Enter or paste news content..."
)

# Prediction Button
if st.button("🔍 Analyze News"):

    if text.strip() == "":
        st.warning("⚠ Please enter some news text.")
    else:

        cleaned_text = clean_text(text)

        vec = vectorizer.transform([cleaned_text])

        pred = model.predict(vec)
        confidence = model.predict_proba(vec)

        confidence_percent = max(confidence[0]) * 100

        st.subheader("📊 Prediction Result")

        if pred[0] == 0:
            st.error("🚨 FAKE NEWS DETECTED")
        else:
            st.success("✅ REAL NEWS DETECTED")

        st.write(f"### Confidence: {confidence_percent:.2f}%")

        st.progress(int(confidence_percent))

# Footer
st.markdown("---")
st.caption("Fake News Detection System | NTCC PROJECT-Anwesha")