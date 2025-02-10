import streamlit as st
import joblib
import nltk  # NLTK for various NLP tasks
from nltk.tokenize import word_tokenize  # Tokenizer to split text into words
from nltk.corpus import stopwords  # List of common stopwords to remove
from nltk.stem import PorterStemmer  # to reduce words to their base form
from nltk import pos_tag  # For part-of-speech tagging

def cleaned_text(text):
    """
    Preprocess the input text:
    1. Convert text to lowercase.
    2. Remove special characters using regex.
    3. Tokenize the cleaned text into words.
    4. Remove common stopwords.
    5. Lemmatize words to get their base form.
    """
    # Step 1: Lowercasing the text
    text = text.lower()

    # Step 3: Tokenize the text (split into words)
    words = word_tokenize(text)

    # Removing special characters
    filter_text = [word for word in words if word.isalnum()]

    # Step 4: Remove stopwords (e.g., "the", "a", "and")
    stop_words = set(stopwords.words('english'))  # Load stopwords
    filtered_words = [word for word in filter_text if word not in stop_words]

    

    # Step 5: Lemmatize words (convert to their root/base form)
    ps = PorterStemmer()
    final_words = [ps.stem(word) for word in filtered_words]

    return " ".join(final_words)

tfidf = joblib.load('vectorizer.pkl','r')
model = joblib.load('model.pkl','r')

st.title('SMS Spam Classifier')
input_sms = st.text_area('Enter the message')

if st.button('Predict'):

    # preprocess
    cleaned_sms = cleaned_text(input_sms)
    # vectorize
    vector_input = tfidf.transform([cleaned_sms])
    # predict 
    result = model.predict(vector_input)[0]
    # display
    if result == 1:
        st.header("Spam")

    else:
        st.header("Not Spam")