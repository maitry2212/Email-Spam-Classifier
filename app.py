import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# App title
st.title("📧 Email Spam Classifier")
st.write("This app uses machine learning to classify messages as spam or not spam.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("SMSSpamCollection", sep='\t', names=["label","message"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df

df = load_data()

# Show dataset preview
if st.checkbox("Show raw dataset"):
    st.write(df.head())

# Split data
X = df["message"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Model Accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"Model trained with {accuracy:.2%} accuracy.")

# Predict user input
st.header("📨 Test a Message")
user_input = st.text_area("Enter a message to classify", "")

if st.button("Predict"):
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    if prediction == 1:
        st.error("🛑 This message is **SPAM**.")
    else:
        st.success("✅ This message is **NOT SPAM**.")


