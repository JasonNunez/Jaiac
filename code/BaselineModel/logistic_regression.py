#!/usr/bin/env python3
"""
Text classification script using logistic regression and TF-Idataframe vectorization.

This script loads and cleans a dataset of text, then evaluates various configurations
of a machine learning pipeline using logistic regression and TF-Idataframe vectorization
for text classification. It utilizes scikit-learn for building and evaluating the
model through cross-validation.
"""

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline

from prepare_data import load_and_clean_data


__author__ = "Jason Nunez"
__version__ = "Spring 2024"
__pylint__ = "2.14.5"


def run_pipeline(configuration, data, labels, cross_validator):
    """
    Runs the pipeline for a given configuration on the provided dataset.

    Args:
        configuration (dict): Configuration settings for the TF-Idataframe vectorizer
        and logistic regression classifier.
        data (pandas.Series): The text data to classify.
        labels (pandas.Series): The labels for the text data.
        cross_validator (kfoldold): The cross-validation splitting strategy.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            binary=configuration['tfidf__binary'],
            max_features=configuration['tfidf__max_features'],
            ngram_range=configuration['tfidf__ngram_range'],
            stop_words=configuration['tfidf__stop_words'],
            use_idf=configuration['tfidf__use_idf']
        )),
        ('clf', LogisticRegression(max_iter=configuration['clf__max_iter']))
    ])

    start_time = time.time()
    accuracies = cross_val_score(pipeline, data, labels, cv=cross_validator, scoring='accuracy')
    end_time = time.time()

    print(f"Running configuration: {configuration}")
    print(f"Accuracies across folds: {accuracies}")
    print(f"Average accuracy: {np.mean(accuracies)}")
    print(f"Time taken: {end_time - start_time} seconds")
    print("---------------------------------------------------------")


def main():
    """
    Main function to load data, define cross-validator, and run configurations.
    """
    dataframe = load_and_clean_data('dev.csv')
    configurations = {
        # Config 1: 0.928%
        1: {
            'clf__max_iter': 100,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': True
        },
        # Config 2: 0.927%
        2: {
            'clf__max_iter': 100,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': False
        },
        # Config 3: 0.927%
        3: {
            'clf__max_iter': 500,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': True
        },
        # Config 4: 0.927%
        4: {
            'clf__max_iter': 1000,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': True
        },
        # Config 5: 0.927%
        5: {
            'clf__max_iter': 2000,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': True
        }
    }

    x_text = dataframe['text']
    y_labels = dataframe['label']

    kfold = KFold(n_splits=5, shuffle=True, random_state=3270)

    for config_number, config_params in configurations.items():
        print(f"Running configuration number {config_number}")
        run_pipeline(config_params, x_text, y_labels, kfold)

if __name__ == '__main__':
    main()
