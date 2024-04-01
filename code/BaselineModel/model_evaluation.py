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
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline

from prepare_data import load_and_clean_data

__author__ = "Jason Nunez"
__version__ = "Spring 2024"
__pylint__ = "2.14.5"


def train_and_evaluate_model_log(configuration, train_data, train_labels, test_data, test_labels):
    """
    Trains the model on the "dev" dataset and evaluates it on the "test" dataset.

    Args:
        configuration (dict): Configuration for the TF-IDF vectorizer and logistic regression classifier.
        train_data (pandas.Series): The text data to train on.
        train_labels (pandas.Series): The labels for the training data.
        test_data (pandas.Series): The text data to test on.
        test_labels (pandas.Series): The labels for the testing data.
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
    pipeline.fit(train_data, train_labels)
    predictions = pipeline.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    end_time = time.time()

    print(f"Running configuration: {configuration}")
    print(f"Test accuracy: {accuracy}")
    print(f"Time taken: {end_time - start_time} seconds")
    print("---------------------------------------------------------")


def train_and_evaluate_model_cat(configuration, train_data, train_labels, test_data, test_labels):
    """
    Trains the model with CatBoost on the training dataset and evaluates it on the testing dataset.

    Args:
        configuration (dict): Configuration for the TF-IDF vectorizer and CatBoost classifier.
        train_data (pandas.Series): The text data to train on.
        train_labels (pandas.Series): The labels for the training data.
        test_data (pandas.Series): The text data to test on.
        test_labels (pandas.Series): The labels for the testing data.
    """
    # Setup the pipeline with TfidfVectorizer and CatBoostClassifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            binary=configuration.get('tfidf__binary', True),
            max_features=configuration.get('tfidf__max_features', None),
            ngram_range=configuration.get('tfidf__ngram_range', (1, 1)),
            stop_words=configuration.get('tfidf__stop_words', 'english'),
            use_idf=configuration.get('tfidf__use_idf', True)
        )),
        ('catboost', CatBoostClassifier(
            iterations=configuration.get('catboost__iterations', 1000),
            learning_rate=configuration.get('catboost__learning_rate', 0.2),
            depth=configuration.get('catboost__depth', 6),
            verbose=0,
            task_type='GPU' if configuration.get('catboost__use_gpu', False) else 'CPU'
        ))
    ])

    start_time = time.time()
    # Train the model on the training data
    pipeline.fit(train_data, train_labels)
    # Predict on the testing data
    predictions = pipeline.predict(test_data)
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    end_time = time.time()

    print(f"Running configuration: {configuration}")
    print(f"Test accuracy: {accuracy}")
    print(f"Time taken: {end_time - start_time} seconds")
    print("---------------------------------------------------------")


def main():
    """
    Main function to load data, define cross-validator, and run configurations.
    """
    # Load training data
    train_df = load_and_clean_data('dev.csv')
    train_x = train_df['text']
    train_y = train_df['label']

    # Load testing data
    test_df = load_and_clean_data('test.csv')
    test_x = test_df['text']
    test_y = test_df['label']

    log_configurations = {
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

    catboost_configurations = {
        1: {
            'catboost__depth': 10,
            'catboost__iterations': 2000,
            'catboost__learning_rate': 0.2,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 1),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': False
        },
        2: {
            'catboost__depth': 10,
            'catboost__iterations': 2000,
            'catboost__learning_rate': 0.2,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 1),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': True
        },
        3: {
            'catboost__depth': 6,
            'catboost__iterations': 2000,
            'catboost__learning_rate': 0.2,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 1),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': True
        },
        4: {
            'catboost__depth': 6,
            'catboost__iterations': 2000,
            'catboost__learning_rate': 0.2,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 1),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': False
        },
        5: {
            'catboost__depth': 10,
            'catboost__iterations': 2000,
            'catboost__learning_rate': 0.1,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 1),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': True
        }
    }

    for config_number, config_params in log_configurations.items():
        print(f"Running configuration number {config_number}")
        train_and_evaluate_model_log(config_params, train_x, train_y, test_x, test_y)

    for config_number, config_params in catboost_configurations.items():
        print(f"Running configuration number {config_number}")
        train_and_evaluate_model_cat(config_params, train_x, train_y, test_x)


if __name__ == '__main__':
    main()
