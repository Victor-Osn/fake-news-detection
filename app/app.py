import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fake News Detection App", page_icon="📰")

# Training data
data = {
    "text": [
        "The government confirmed a new education policy today.",
        "Scientists discovered water on Mars according to a new report.",
        "A celebrity was secretly replaced by a robot last year.",
        "Doctors recommend drinking enough water daily for good health.",
        "Aliens have opened a shopping mall in New York.",
        "The central bank announced a change in interest rates.",
        "Eating fruits and vegetables supports a healthy lifestyle.",
        "A man claims his dog can predict the stock market.",
        "Researchers developed a new method for early disease detection.",
        "A fake miracle cure can heal every disease instantly."
    ],
    "label": ["REAL", "REAL", "FAKE", "REAL", "FAKE", "REAL", "REAL", "FAKE", "REAL", "FAKE"]
}

df = pd.DataFrame(data)

# Convert labels into numbers
df["label_num"] = df["label"].map({"REAL": 1, "FAKE": 0})

# Split features and target
X = df["text"]
y = df["label_num"]

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Reusable prediction function
def predict_news(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return "REAL" if prediction == 1 else "FAKE"

st.title("Fake News Detection Web App")
st.write("Enter a news statement below and the app will predict whether it is REAL or FAKE.")

news_text = st.text_area("Enter news text here:")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter a news statement.")
    else:
        result = predict_news(news_text)
        if result == "REAL":
            st.success(f"Prediction: {result}")
        else:
            st.error(f"Prediction: {result}")