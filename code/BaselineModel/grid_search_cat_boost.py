#!/usr/bin/env python3
"""
Grid Search Cat Boost model for emotional text classification.
"""
import time
import pandas as pd


from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold
from prepare_data import load_and_clean_data

__author__ = 'Jason Nunez'
__version__ = 'Spring 2024'
__pylint__= '2.14.5'

def main():
    """
    Main function to load data, define cross-validator, and run configurations.
    """
    # Set option to display more rows
    pd.set_option('display.max_rows', 500)

    # Set option to display more columns
    pd.set_option('display.max_columns', 50)

    pd.set_option('display.max_colwidth', None)

    dataframe = load_and_clean_data('text.csv')

    # Define your text and labels
    x_text = dataframe['text']
    y_labels = dataframe['label']

    # Set up the pipeline
    pipeline = Pipeline([
        ('tfidataframe', TfidfVectorizer(stop_words='english')),
        ('catboost', CatBoostClassifier(task_type='GPU', verbose=0))
    ])

    # Define the parameter grid
    param_grid = {
        'tfidf__max_features': [2000],
        'tfidf__binary': [True],
        'tfidf__use_idf': [True, False],
        'tfidf__stop_words': ['english'],
        'tfidf__ngram_range': [(1, 1)],
        'catboost__iterations': [100, 500, 1000, 2000],  # Number of trees.
        'catboost__learning_rate': [0.01, 0.05, 0.1, 0.2],  # Used to prevent overfitting.
        'catboost__depth': [6, 10, 20, 30],  # Depth of each tree.
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=3270)

    # Set up GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=kfold, verbose=3)

    # Track the start time
    start_time = time.time()

    # Execute the grid search
    grid_search.fit(x_text, y_labels)

    # Calculate the duration
    duration = time.time() - start_time

    # If you want the top 5 configurations:
    results = pd.DataFrame(grid_search.cv_results_)
    top5 = results.nlargest(5, 'mean_test_score')
    print(f"Time taken: {duration} seconds")
    print(top5[['params', 'mean_test_score', 'rank_test_score']])

if __name__ == "__main__":
    main()
    