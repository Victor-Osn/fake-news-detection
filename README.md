# Fake News Detection Web App

A machine learning project that classifies news statements as REAL or FAKE using natural language processing techniques.

## Project Overview

This project demonstrates a fake news detection workflow using a small labeled dataset created in Google Colab. The notebook prepares text data, converts labels into numeric form, transforms text with TF-IDF, trains a Logistic Regression model, evaluates performance, and predicts whether custom news statements are real or fake.

## Features

- Text classification for fake news detection
- Label encoding for machine learning
- Train and test split
- TF-IDF text vectorization
- Logistic Regression model training
- Accuracy and classification report
- Custom news prediction function

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Google Colab

## Project File

- fake_news_detection.ipynb - complete notebook with code, outputs, and project summary

## Current Status

Completed machine learning project foundation with prediction workflow and reusable prediction function.

## Web App

This project also includes a Streamlit web interface inside the app folder. Users can enter a news statement and get a prediction showing whether the text is likely REAL or FAKE.

## How to Run

# Streamlit App

1. Open terminal in the project folder
2. Run py -m pip install -r requirements.txt
3. Run py -m streamlit run app/app.py
4. Open the local Streamlit URL in the browser

## Future Improvements

- Use a larger real-world dataset
- Improve model reliability
- Build a web interface for user input
- Deploy as a web application
