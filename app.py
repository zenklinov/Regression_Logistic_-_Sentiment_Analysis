import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from matplotlib.cm import viridis_r
from matplotlib.colors import Normalize
from nltk.stem import PorterStemmer
import requests
from io import BytesIO

# NLTK resources download
nltk.download('punkt')
nltk.download('stopwords')

# Function to load joblib file from a URL
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        model = load(BytesIO(response.content))
        return model
    else:
        raise Exception(f"Failed to load model from {url}")

# Load models from GitHub URLs
tfidf_url = "https://raw.githubusercontent.com/zenklinov/Regression_Logistic_-_Sentiment_Analysis_Movie_Data/main/tfidf_vectorizer.joblib"
lr_url = "https://raw.githubusercontent.com/zenklinov/Regression_Logistic_-_Sentiment_Analysis_Movie_Data/main/logistic_regression_model.joblib"

# Load the models
tfidf = load_model_from_url(tfidf_url)
lr = load_model_from_url(lr_url)

# Define tokenizer function using Porter Stemmer
def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in word_tokenize(text)]

# Preprocessing function
def clean_text(text):
    text = re.sub(r'@\w+|#\w+|http\S+|www\S+|[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join(word for word in words if word not in stop_words)
    return cleaned_text

# Streamlit app
st.title("Sentiment Analysis App")

# File upload
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Select column for sentiment analysis
    text_column = st.selectbox("Select the column for sentiment analysis:", df.columns)

    if text_column:
        # Preprocess text
        st.subheader("Preprocessing")
        df['cleaned_text'] = df[text_column].apply(clean_text)
        st.write("Preview of preprocessed data:")
        st.dataframe(df[[text_column, 'cleaned_text']].head())

        # Word Cloud
        st.subheader("Word Cloud")
        all_text = ' '.join(df['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud")
        st.pyplot(plt)

        # Top word frequency
        st.subheader("Top Word Frequency")
        all_words = all_text.split()
        word_counts = Counter(all_words)
        word_freq_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
        top_words = word_freq_df.head(20)
        st.write(top_words)

        norm = Normalize(vmin=min(top_words['Frequency']), vmax=max(top_words['Frequency']))
        colors = [viridis_r(norm(value)) for value in top_words['Frequency']]
        plt.figure(figsize=(10, 6))
        plt.bar(top_words['Word'], top_words['Frequency'], color=colors)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 20 Most Frequent Words')
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Sentiment Prediction
        st.subheader("Sentiment Prediction")
        all_data_tfidf = tfidf.transform(df['cleaned_text'])
        df['predicted_sentiment'] = lr.predict(all_data_tfidf)
        st.write("Sentiment Prediction Results:")
        st.dataframe(df[['cleaned_text', 'predicted_sentiment']])

        # Sentiment visualization
        sentiment_counts = df['predicted_sentiment'].value_counts()
        negative_count = sentiment_counts.get(0, 0)
        positive_count = sentiment_counts.get(1, 0)

        labels = [f'Negative (0): {negative_count}', f'Positive (1): {positive_count}']
        sizes = [negative_count, positive_count]
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['red', 'green'])
        plt.title('Sentiment Distribution')
        plt.axis('equal')
        st.pyplot(plt)

        st.write(f"Negative Sentiment: {negative_count}")
        st.write(f"Positive Sentiment: {positive_count}")