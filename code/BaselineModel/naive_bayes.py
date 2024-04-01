#!/usr/bin/env python3
"""
Multinomial Naïve Bayes model for emotional text classification.

"""

import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from prepare_data import load_and_clean_data

__author__ = 'Alec Neal'
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

    kf = KFold(n_splits=5, shuffle=True, random_state=3270)

    pl = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    param_grid = {
        'vectorizer__max_features': [None, 500, 1000, 2000],
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vectorizer__stop_words': [None, 'english'],
        'classifier__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'classifier__fit_prior': [True, False]
    }

    grid_search = GridSearchCV(pl, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=4)

    start_time = time.time()
    grid_search.fit(X, y)
    end_time = time.time()

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_}")
    print(f"Time taken for GridSearchCV: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
    