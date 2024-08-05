import streamlit as st
import nltk
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the trained model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_review_sentiment(review):
    processed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([processed_review]).toarray()
    prediction = model.predict(vectorized_review)
    return 'Liked' if prediction[0] == 1 else 'Disliked'

# Streamlit UI
st.title('Review Sentiment Analysis')

st.write("Enter a review below and get its sentiment prediction:")

user_input = st.text_area("Review")

if st.button('Predict Sentiment'):
    if user_input:
        sentiment = predict_review_sentiment(user_input)
        sentiment_colored = f'<span style="color: {"green" if sentiment == "Liked" else "red"};">{sentiment}</span>'
        st.markdown(f'Sentiment: {sentiment_colored}', unsafe_allow_html=True)
    else:
        st.write("Please enter a review to predict sentiment.")
