import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from io import BytesIO
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.stem import PorterStemmer

# NLTK resources download
nltk.download('punkt')
nltk.download('stopwords')

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

# Function to download file from GitHub
def download_file(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise error if request fails
    return BytesIO(response.content)

# Streamlit app
st.title("Sentiment Analysis App")

# Creator information
st.write("Creator: Amanatullah Pandu Zenklinov")
st.markdown("""
[LinkedIn](https://www.linkedin.com/in/zenklinov/) | 
[GitHub](https://github.com/zenklinov/) | 
[Instagram](https://instagram.com/zenklinov)
""")

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

    # Preprocessing button
    if st.button("Preprocess Text"):
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
            st.image(wordcloud.to_array(), use_column_width=True)

            # Top word frequency
            st.subheader("Top Word Frequency")
            all_words = all_text.split()
            word_counts = Counter(all_words)
            word_freq_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
            top_words = word_freq_df.head(20)
            st.write(top_words)

            # Interactive bar chart for top words frequency using Plotly
            fig = go.Figure(data=[go.Bar(
                x=top_words['Word'],
                y=top_words['Frequency'],
                marker=dict(color=top_words['Frequency'], colorscale='Viridis')
            )])
            fig.update_layout(
                title="Top 20 Most Frequent Words",
                xaxis_title="Words",
                yaxis_title="Frequency",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig)

            # URLs for TF-IDF and Logistic Regression Model on GitHub
            url_tfidf = "https://github.com/zenklinov/Regression_Logistic_-_Sentiment_Analysis_Movie_Data/raw/main/tfidf_vectorizer.joblib"
            url_lr_model = "https://github.com/zenklinov/Regression_Logistic_-_Sentiment_Analysis_Movie_Data/raw/main/logistic_regression_model.joblib"

            # Download the files
            tfidf_file = download_file(url_tfidf)
            lr_model_file = download_file(url_lr_model)

            # Load TF-IDF and model
            tfidf = load(tfidf_file)
            lr = load(lr_model_file)

            # Predict sentiment
            st.subheader("Sentiment Prediction")
            all_data_tfidf = tfidf.transform(df['cleaned_text'])
            df['predicted_sentiment'] = lr.predict(all_data_tfidf)
            st.write("Sentiment Prediction Results:")
            st.dataframe(df[['cleaned_text', 'predicted_sentiment']])

            # Interactive pie chart for sentiment distribution using Plotly
            sentiment_counts = df['predicted_sentiment'].value_counts()
            negative_count = sentiment_counts.get(0, 0)
            positive_count = sentiment_counts.get(1, 0)

            labels = [f'Negative (0): {negative_count}', f'Positive (1): {positive_count}']
            sizes = [negative_count, positive_count]

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=sizes,
                marker=dict(colors=['red', 'green']),
                hoverinfo='label+percent',
                textinfo='value+percent'
            )])
            fig.update_layout(title='Sentiment Distribution')
            st.plotly_chart(fig)

            st.write(f"Negative Sentiment: {negative_count}")
            st.write(f"Positive Sentiment: {positive_count}")
