import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

nltk.download('stopwords')
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- Page setup ---
st.set_page_config(page_title="Sentiment Analyzer 💬", layout="wide")
st.title("💬 Sentiment Analysis Web App")
st.write("Enter a sentence or review below to analyze its sentiment!")

# --- Sample dataset ---
data = {
    'text': [
        'I love this movie, it was fantastic!',
        'This is the worst product I have ever bought.',
        'It was an average experience, nothing special.',
        'The food was delicious and service was great!',
        'I am disappointed with the quality.',
        'Totally worth it, I will recommend to everyone!',
        'It’s not bad but could be better.',
        'I hate this app, it’s so buggy!',
        'This phone works perfectly fine, very smooth.',
        'The experience was terrible, not coming again.'
    ],
    'label': ['positive', 'negative', 'neutral', 'positive', 'negative',
              'positive', 'neutral', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

# --- Text preprocessing ---
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(preprocess_text)

# --- Feature extraction ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model training ---
model = MultinomialNB()
model.fit(X_train, y_train)

# --- Evaluate accuracy ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- User Input ---
st.subheader("📝 Enter text for sentiment prediction:")
user_input = st.text_area("Type here:", height=100)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([clean_input])
        prediction = model.predict(input_vector)[0]
        prob = model.predict_proba(input_vector)[0]
        
        st.markdown(f"### 🧠 Sentiment: **{prediction.upper()}**")
        st.write(f"Model Accuracy: `{acc*100:.2f}%`")

        # Show probabilities chart
        fig, ax = plt.subplots()
        ax.bar(model.classes_, prob, color=['green','red','gray'])
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Confidence")
        ax.set_title("Sentiment Probability")
        st.pyplot(fig)

        # Add emoji
        if prediction == "positive":
            st.markdown("😊 Great vibes detected!")
        elif prediction == "negative":
            st.markdown("😞 Seems negative!")
        else:
            st.markdown("😐 Neutral opinion detected.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & NLP")
