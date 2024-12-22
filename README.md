# Sentiment Analysis with Logistic Regression

This project demonstrates sentiment analysis, model based on movie review data using Logistic Regression. The model predicts whether a review expresses positive or negative sentiment based on the text provided. The application on #PepGuardiola from Twitter (X).

## Project Structure

- `classify_sentiment.ipynb`: Jupyter Notebook of Input Data, Preprocessing, until Visualizing Sentiment Analysis using Logistic Regression.
- `app.py`: Main Streamlit application for running the sentiment analysis interface.
- `logistic_regression_model.joblib`: Pre-trained Logistic Regression model.
- `tfidf_vectorizer.joblib`: TF-IDF vectorizer for text preprocessing.

## Features

- Upload or input text to analyze sentiment.
- Pre-trained model for high-accuracy predictions.
- Real-time sentiment prediction.

## Requirements

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Pre-trained Model and Vectorizer

The Logistic Regression model and TF-IDF vectorizer used in this project are pre-trained and available at the following locations:

- [Logistic Regression Model](https://github.com/zenklinov/Regression_Logistic_-_Sentiment_Analysis_Movie_Data/blob/main/logistic_regression_model.joblib)
- [TF-IDF Vectorizer](https://github.com/zenklinov/Regression_Logistic_-_Sentiment_Analysis_Movie_Data/blob/main/tfidf_vectorizer.joblib)

Ensure these files are downloaded and placed in the appropriate directory before running the application.

## Usage

Run the Streamlit application with the following command:

```bash
streamlit run app.py
```

## Try it on Streamlit:

https://regressionlogistic-sentimentanalysis-lgbxfzczme5clpzfct9qmg.streamlit.app/
