# 📧 Email Spam Classifier using Machine Learning

An interactive **Streamlit web app** that classifies email messages as **Spam** or **Not Spam** using **Natural Language Processing (NLP)** and **Machine Learning**.

This project uses a TF-IDF vectorizer and a Naive Bayes classifier trained on the famous [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). It offers a simple UI to test custom messages in real-time.

---

## 💡 Features

- ✅ Train and evaluate a machine learning model in the background
- 📬 Classify messages as "Spam" or "Not Spam"
- 🧠 TF-IDF + Naive Bayes model (95%+ accuracy)
- 📊 Toggle raw dataset preview
- ⚡ Real-time predictions via Streamlit

---

## 🧠 How It Works

1. **Dataset Preprocessing**: Clean and label the dataset
2. **Text Vectorization**: Use `TfidfVectorizer` to convert text to numeric form
3. **Model Training**: Train a `MultinomialNB` classifier
4. **Prediction**: Classify user input messages instantly
5. **Evaluation**: Show model accuracy and metrics

---

## Install Dependencies
pip install -r requirements.txt

---

## run the app
streamlit run app.py

---

⭐ Support

If you find this project helpful, give it a ⭐ on GitHub and share it with others! Contributions are welcome 😊


