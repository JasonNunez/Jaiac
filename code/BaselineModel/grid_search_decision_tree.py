#!/usr/bin/env python3
"""
Decision Tree model for emotional text classification.
"""
import time
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from prepare_data import load_and_clean_data

__author__ = 'NAME'
__version__ = 'Spring 2024'
__pylint__= '2.14.5'

# pylint: disable=C0103

def main():
    """
    Main function to load data, define cross-validator, and run configurations.
    """
    # Set option to display more rows
    pd.set_option('display.max_rows', 500)

    # Set option to display more columns
    pd.set_option('display.max_columns', 50)

    pd.set_option('display.max_colwidth', None)

    df = load_and_clean_data('text.csv')

    # Define your text and labels
    X = df['text']
    y = df['label']

    # Set up the pipeline
    pl = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('decisiontree', DecisionTreeClassifier())
    ])

    # Define the parameter grid
    param_grid = {
        'tfidf__max_features': [2000],
        'tfidf__binary': [True],
        'tfidf__use_idf': [True, False],
        'tfidf__stop_words': ['english'],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'decisiontree__max_depth': [10, 50, 100, 200],
        'decisiontree__min_samples_split': [2, 50, 100],
        'decisiontree__min_samples_leaf': [1, 20, 50],
        'decisiontree__class_weight': [None, 'balanced']
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=3270)

    # Set up GridSearchCV
    grid_search = GridSearchCV(pl, param_grid, scoring='accuracy', cv=kfold, verbose=3, n_jobs=-1)

    # Track the start time
    start_time = time.time()

    # Execute the grid search
    grid_search.fit(X, y)

    # Calculate the duration
    duration = time.time() - start_time

    # Print the best parameters and their corresponding accuracy
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best accuracy: {grid_search.best_score_}")
    print(f"Time taken: {duration} seconds")

    # If you want the top 5 configurations:
    results = pd.DataFrame(grid_search.cv_results_)
    top5 = results.nlargest(5, 'mean_test_score')
    print(top5[['params', 'mean_test_score', 'rank_test_score']])

if __name__ == "__main__":
    main()
    