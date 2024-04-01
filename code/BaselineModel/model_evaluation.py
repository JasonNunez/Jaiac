#!/usr/bin/env python3
"""
Text classification script using logistic regression and TF-Idataframe vectorization.

This script loads and cleans a dataset of text, then evaluates various configurations
of a machine learning pipeline using logistic regression and TF-Idataframe vectorization
for text classification. It utilizes scikit-learn for building and evaluating the
model through cross-validation.
"""

import time
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

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
            task_type='GPU' if configuration.get('catboost__use_gpu', False) else 'CPU'
        ))
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


def train_and_evaluate_model_decision(configuration, train_data, train_labels, test_data, test_labels):
    """
    Trains the model with a Decision Tree on the training dataset and evaluates it on the testing dataset.

    Args:
        configuration (dict): Configuration for the TF-IDF vectorizer and Decision Tree classifier.
        train_data (pandas.Series): The text data to train on.
        train_labels (pandas.Series): The labels for the training data.
        test_data (pandas.Series): The text data to test on.
        test_labels (pandas.Series): The labels for the testing data.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            binary=configuration.get('tfidf__binary', True),
            max_features=configuration.get('tfidf__max_features', None),
            ngram_range=configuration.get('tfidf__ngram_range', (1, 1)),
            stop_words=configuration.get('tfidf__stop_words', 'english'),
            use_idf=configuration.get('tfidf__use_idf', True)
        )),
        ('decision_tree', DecisionTreeClassifier(
            max_depth=configuration.get('decisiontree__max_depth', None),
            min_samples_split=configuration.get('decisiontree__min_samples_split', 2),
            min_samples_leaf=configuration.get('decisiontree__min_samples_leaf', 1),
            class_weight=configuration.get('decisiontree__class_weight', None)
        ))
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


def train_and_evaluate_model_random(config_params, train_data, train_labels, test_data, test_labels):
    """
        Trains the model with a Random Forest on the training dataset and evaluates it on the testing dataset.

        Args:
            config_params (dict): Configuration for the TF-IDF vectorizer and Random Forest classifier.
            train_data (pandas.Series): The text data to train on.
            train_labels (pandas.Series): The labels for the training data.
            test_data (pandas.Series): The text data to test on.
            test_labels (pandas.Series): The labels for the testing data.
        """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            binary=config_params.get('tfidf__binary', True),
            max_features=config_params.get('tfidf__max_features', None),
            ngram_range=config_params.get('tfidf__ngram_range', (1, 1)),
            stop_words=config_params.get('tfidf__stop_words', 'english'),
            use_idf=config_params.get('tfidf__use_idf', True)
        )),
        ('random_forest', RandomForestClassifier(
            n_estimators=config_params.get('rf__n_estimators', 100),
            max_depth=config_params.get('rf__max_depth', None),
            min_samples_split=config_params.get('rf__min_samples_split', 2),
            min_samples_leaf=config_params.get('rf__min_samples_leaf', 1),
            max_features=config_params.get('rf__max_features', 'auto'),
            class_weight=config_params.get('rf__class_weight', None)
        ))
    ])

    start_time = time.time()
    pipeline.fit(train_data, train_labels)
    predictions = pipeline.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    end_time = time.time()

    print(f"Running configuration: {config_params}")
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
        1: {
            'clf__max_iter': 100,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': True
        },
        2: {
            'clf__max_iter': 100,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': False
        },
        3: {
            'clf__max_iter': 500,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': True
        },
        4: {
            'clf__max_iter': 1000,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': True
        },
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

    decisiontree_configurations = {
        1: {
            'decisiontree__class_weight': None,
            'decisiontree__max_depth': 200,
            'decisiontree__min_samples_leaf': 20,
            'decisiontree__min_samples_split': 100,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 3),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': False
        },
        2: {
            'decisiontree__class_weight': None,
            'decisiontree__max_depth': 200,
            'decisiontree__min_samples_leaf': 20,
            'decisiontree__min_samples_split': 100,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': False
        },
        3: {
            'decisiontree__class_weight': None,
            'decisiontree__max_depth': 200,
            'decisiontree__min_samples_leaf': 50,
            'decisiontree__min_samples_split': 100,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': False
        },
        4: {
            'decisiontree__class_weight': None,
            'decisiontree__max_depth': 200,
            'decisiontree__min_samples_leaf': 2,
            'decisiontree__min_samples_split': 100,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 2),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': False
        },
        5: {
            'decisiontree__class_weight': None,
            'decisiontree__max_depth': 200,
            'decisiontree__min_samples_leaf': 100,
            'decisiontree__min_samples_split': 100,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 3),
            'tfidf__stop_words': 'english',
            'tfidf__use_idf': False
        }
    }

    random_forest_configurations = {
        1: {
            'rf__max_depth': 200,
            'rf__max_features': 'sqrt',
            'rf__min_samples_leaf': 10,
            'rf__min_samples_split': 50,
            'rf__n_estimators': 300,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 1),
            'tfidf__use_idf': False,
        },
        2: {
            'rf__max_depth': 200,
            'rf__max_features': 'sqrt',
            'rf__min_samples_leaf': 10,
            'rf__min_samples_split': 50,
            'rf__n_estimators': 300,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 1),
            'tfidf__use_idf': True,
        },
        3: {
            'rf__max_depth': 100,
            'rf__max_features': 'log2',
            'rf__min_samples_leaf': 50,
            'rf__min_samples_split': 100,
            'rf__n_estimators': 300,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 1),
            'tfidf__use_idf': False,
        },
        4: {
            'rf__max_depth': 100,
            'rf__max_features': 'log2',
            'rf__min_samples_leaf': 50,
            'rf__min_samples_split': 100,
            'rf__n_estimators': 300,
            'tfidf__binary': False,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 1),
            'tfidf__use_idf': False,
        },
        5: {
            'rf__max_depth': 100,
            'rf__max_features': 'log2',
            'rf__min_samples_leaf': 50,
            'rf__min_samples_split': 100,
            'rf__n_estimators': 100,
            'tfidf__binary': True,
            'tfidf__max_features': 2000,
            'tfidf__ngram_range': (1, 1),
            'tfidf__use_idf': False,
        }
    }

    for config_number, config_params in log_configurations.items():
        print(f"Running configuration number {config_number}")
        train_and_evaluate_model_log(config_params, train_x, train_y, test_x, test_y)

    for config_number, config_params in catboost_configurations.items():
        print(f"Running configuration number {config_number}")
        train_and_evaluate_model_cat(config_params, train_x, train_y, test_x, test_y)

    for config_number, config_params in decisiontree_configurations.items():
        print(f"Running configuration number {config_number}")
        train_and_evaluate_model_decision(config_params, train_x, train_y, test_x, test_y)

    for config_number, config_params in random_forest_configurations.items():
        print(f"Running configuration number {config_number}")
        train_and_evaluate_model_random(config_params, train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()
