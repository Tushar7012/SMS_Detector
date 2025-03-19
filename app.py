# Importing Dependencies
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Download NLTK resources if not available
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Check if model and vectorizer exist
if not os.path.exists("vectorizer.pkl") or not os.path.exists("model.pkl"):
    st.error("Error: Model or vectorizer file is missing!")
else:
    # Load the Vectorizer and Model
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    mnb = pickle.load(open("model.pkl", "rb"))

    # Title of the Application
    st.title("Spam Ham Message Classifier")
    input_sms = st.text_area("Enter the Message")

    if st.button("Predict"):
        transformed_text = transform_text(input_sms)
        vectorized = tfidf.transform([transformed_text])

        result = mnb.predict(vectorized)[0]

        # Debugging info (remove in production)
        st.write(f"Transformed text: {transformed_text}")
        st.write(f"Vectorized shape: {vectorized.shape}")
        st.write(f"Prediction: {result}")

        if result == 1:
            st.header("Spam")
        else:
            st.header("Ham")
