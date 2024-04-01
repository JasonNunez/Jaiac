#!/usr/bin/env python3
"""
Random Forest model for emotional text classification.
"""
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from prepare_data import load_and_clean_data

__author__ = "Liam Cole"
__version__ = "Spring 2024"
__pylint__ = "2.14.5"

def main():
    """
    Main function to load data, define cross-validator, and run configurations.
    """
    dataframe = load_and_clean_data('text.csv')

    x_text = dataframe['text']
    y_labels = dataframe['label']

    # Set up the pipeline
    pln = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('rf', RandomForestClassifier())
    ])

    # Define the parameter grid
    param_grid = {
        'tfidf__max_features': [2000],
        'tfidf__binary': [True],
        'tfidf__use_idf': [True, False],
        'tfidf__ngram_range': [(1, 1)],
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [10, 50, 100, 200],
        'rf__min_samples_split': [10, 50, 100],
        'rf__min_samples_leaf': [10, 20, 50],
        'rf__max_features': ['sqrt', 'log2']
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=3270)

    # Set up GridSearchCV
    grid_search = GridSearchCV(pln, param_grid, scoring='accuracy', cv=kfold, verbose=3, n_jobs=-1)

    # Track the start time
    start_time = time.time()

    # Execute the grid search
    grid_search.fit(x_text, y_labels)

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

if __name__ == '__main__':
    main()
    