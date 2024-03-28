#!/usr/bin/env python3
"""
Multinomial Naïve Bayes.
Note: C0103 to be consistent with sklearn namings.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import time
import re

__author__ = 'Alec Neal'
__version__ = 'Spring 2024'
__pylint__= '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'

# pylint: disable=C0103

if __name__ == "__main__":
    # Load the dataset
    print("Loading data...")
    data = pd.read_csv('text.csv')
    print("Data loaded.")

    # Preprocess the text
    print("Preparing data...")
    
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text).strip()  # Removes all extra spaces
        text = re.sub(r'(href|http|img|src|span style).*', '', text)  # Removes all words to the right of urls
        text = re.sub(' t ', 't ', text)  # Fixes floating t's (don t -> dont)
        text = re.sub(' s ', 's ', text)  # Fixes floating s's (it s -> its)
        text = re.sub(' i m ', ' im ', text)  # Fixes i m -> im seperation
        text = re.sub(' e mail', ' email', text)  # Fixes e mail -> mail
        text = re.sub(' w e ', ' we ', text)  # Fixes w e -> we seperation
        text = re.sub(' we d ', ' wed ', text)  # Fixes we d -> wed seperation
        text = re.sub(' i d ', ' id ', text)  # Fixes i d -> id seperation
        text = re.sub(' they d ', ' theyd ', text)  # Fixes they d -> theyd seperation
        return text
    
    data['text'] = data['text'].apply(clean_text)

    data = data.drop_duplicates(subset='text')
    data = data.dropna()
    
    print("Text cleaned.")
    
    X = data['text']
    y = data['label']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=3270)

    # Create a pipeline with CountVectorizer and MultinomialNB
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Train the model
    pipeline.fit(X, y)
    
    start_time = time.time()
    accuracies = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
    end_time = time.time()

    # Print the results
    print(f"Accuracies across folds: {accuracies}")
    print(f"Average accuracy: {np.mean(accuracies)}")
    print(f"Time taken for 5-Fold CV: {end_time - start_time} seconds")


    
    