#!/usr/bin/env python3
"""
Decision Tree model for emotional text classification.
"""
import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from prepare_data import load_and_clean_data

__author__ = 'Jason Nunez'
__version__ = 'Spring 2024'
__pylint__= '2.14.5'

# pylint: disable=C0103

def main():
    """
    Main function to load data, define cross-validator, and run configurations.
    """
    df = load_and_clean_data('text.csv')

    X = df['text']
    y = df['label']

    # Define the K-Fold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=3270)

    pipeline = make_pipeline(
        TfidfVectorizer(max_features=500,stop_words='english'),
        DecisionTreeClassifier(max_depth=10, random_state=3270)
    )

    # Measure the time taken by the K-Fold cross-validation process
    start_time = time.time()
    accuracies = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
    end_time = time.time()

    # Print the results
    print(f"Accuracies across folds: {accuracies}")
    print(f"Average accuracy: {np.mean(accuracies)}")
    print(f"Time taken for 5-Fold CV: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
    