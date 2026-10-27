\# Movie Review Sentiment Analysis Using NLP and Machine Learning



\## 1. Project Overview



This project implements a Movie Review Sentiment Analysis system using Natural Language Processing (NLP) and Machine Learning.



The system analyzes a movie review and predicts whether its sentiment is positive or negative.



The project uses the NLTK Movie Reviews dataset for training and evaluation. Multiple machine learning models were compared, and an optimized Linear Support Vector Machine (SVM) using TF-IDF unigram and bigram features was used as the final model.



An interactive Streamlit web application was also developed to allow users to enter their own movie reviews and receive sentiment predictions.



\---



\## 2. Objectives



\- Perform sentiment analysis on movie reviews.

\- Apply NLP techniques for text preprocessing.

\- Convert textual data into numerical features using TF-IDF.

\- Train and compare different machine learning classifiers.

\- Optimize a Linear SVM model using unigram and bigram features.

\- Evaluate the final model on a held-out test dataset.

\- Develop an interactive Streamlit application for sentiment prediction.



\---



\## 3. Dataset



The project uses the NLTK Movie Reviews dataset.



The dataset contains:



\- 2,000 movie reviews

\- 1,000 positive reviews

\- 1,000 negative reviews



An 80:20 stratified train-test split was used:



\- Training set: 1,600 reviews

\- Test set: 400 reviews



\---



\## 4. Technologies Used



\- Python

\- NLTK

\- Scikit-learn

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn

\- Joblib

\- Streamlit



\---



\## 5. NLP Preprocessing



The following preprocessing steps were applied:



1\. Convert text to lowercase.

2\. Split the review into words.

3\. Remove non-alphabetic tokens.

4\. Remove English stopwords.

5\. Apply WordNet lemmatization.



The preprocessing implementation is available in:



```text

src/preprocessing.py

