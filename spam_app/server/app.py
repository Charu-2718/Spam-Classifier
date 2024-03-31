import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(_name_)

# Load the spam dataset and preprocess
def load_and_preprocess_data():
    df = pd.read_csv('spam.csv')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])
    df.drop_duplicates(keep='first', inplace=True)
    nltk.download('stopwords')
    nltk.download('punkt')
    return df

# Define PorterStemmer outside the function
ps = PorterStemmer()

# Tokenization and feature extraction
def tokenize_and_extract_features(df):
    df['num_characters'] = df['text'].apply(len)
    df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
    df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

    def transform_text(text):
        stopwords_english = set(stopwords.words('english'))
        text = text.lower()
        text = nltk.word_tokenize(text)
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]
        y.clear()
        for i in text:
            if i not in stopwords_english and i not in string.punctuation:
                y.append(i)
        text = y[:]
        y.clear()
        for i in text:
            y.append(ps.stem(i))
        return " ".join(y)

    df['transformed_text'] = df['text'].apply(transform_text)
    return df

# Generate WordCloud for spam and ham
def generate_wordcloud(df):
    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
    plt.figure(figsize=(15,6))
    plt.imshow(spam_wc)
    ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
    plt.figure(figsize=(15,6))
    plt.imshow(ham_wc)

# Prepare TF-IDF Vectorization
def prepare_tfidf(df):
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['transformed_text']).toarray()
    y = df['target'].values
    return X, y, tfidf

# Train Logistic Regression Classifier
def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    lrc = LogisticRegression(solver='liblinear', penalty='l1')
    lrc.fit(X_train, y_train)
    return lrc, X_train, X_test, y_train, y_test

# Flask route for spam classification
@app.route('/classify_spam', methods=['POST'])
def classify_spam():
    # Get the input sentence from the request
    data = request.get_json()
    sentence = data.get('sentence')

    # Define the transform_text function here
    def transform_text(text):
        stopwords_english = set(stopwords.words('english'))
        text = text.lower()
        text = nltk.word_tokenize(text)
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]
        y.clear()
        for i in text:
            if i not in stopwords_english and i not in string.punctuation:
                y.append(i)
        text = y[:]
        y.clear()
        for i in text:
            y.append(ps.stem(i))
        return " ".join(y)

    # Preprocess the input sentence
    processed_sentence = transform_text(sentence)

    # Vectorize the processed sentence
    vectorized_sentence = tfidf.transform([processed_sentence]).toarray()

    # Predict using the trained logistic regression classifier
    prediction = lrc.predict(vectorized_sentence)

    # Convert the prediction to human-readable form
    prediction_label = 'spam' if prediction[0] == 1 else 'ham'

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction_label})

# Main function to run Flask app
if _name_ == '_main_':
    # Load and preprocess data
    df = load_and_preprocess_data()

    # Tokenize and extract features
    df = tokenize_and_extract_features(df)

    # Generate WordCloud
    generate_wordcloud(df)

    # Prepare TF-IDF Vectorization
    X, y, tfidf = prepare_tfidf(df)

    # Train Logistic Regression Classifier
    lrc, X_train, X_test, y_train, y_test = train_classifier(X, y)

    # Run Flask app
    app.run(debug=True)