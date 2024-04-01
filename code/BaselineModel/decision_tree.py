#!/usr/bin/env python3
"""
Decision Tree model for emotional text classification.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from prepare_data import load_and_clean_data

__author__ = 'Jason Nunez'
__version__ = 'Spring 2024'
__pylint__= '2.14.5'

def run_pipeline(configuration, data, labels, cross_validator):
    """
    Runs the pipeline for a given configuration on the provided dataset.
    Returns the average accuracy and configuration details for further processing.

    Args:
        configuration (dict): Configuration settings for the TF-IDF vectorizer and DecisionTreeClassifier.
        data (pandas.Series): The text data to classify.
        labels (pandas.Series): The labels for the text data.
        cross_validator (KFold): The cross-validation splitting strategy.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            binary=configuration['tfidf__binary'],
            max_features=configuration['tfidf__max_features'],
            ngram_range=configuration['tfidf__ngram_range'],
            stop_words=configuration['tfidf__stop_words'],
            use_idf=configuration['tfidf__use_idf']
        )),
        ('decisiontree', DecisionTreeClassifier(
            class_weight=configuration['decisiontree__class_weight'],
            max_depth=configuration['decisiontree__max_depth'],
            min_samples_leaf=configuration['decisiontree__min_samples_leaf'],
            min_samples_split=configuration['decisiontree__min_samples_split'],
            random_state=3270  # Ensuring reproducibility
        ))
    ])

    accuracies = cross_val_score(pipeline, data, labels, cv=cross_validator, scoring='accuracy', n_jobs=-1)
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy, configuration

def main():
    """
    Main function to load data, define cross-validator, and run configurations.
    """
    df = load_and_clean_data('dev.csv')

    configurations = {
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

    X = df['text']
    y = df['label']

    kf = KFold(n_splits=5, shuffle=True, random_state=3270)

    results = []

    for config_number, config_params in configurations.items():
        print(f"Running configuration number {config_number}")
        avg_accuracy, config = run_pipeline(config_params, X, y, kf)
        results.append((avg_accuracy, config))

    # Sort the results by average accuracy in descending order
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)

    # Print summary of top configurations
    print("Top Decision Tree Configurations:")
    for avg_accuracy, config in sorted_results:
        print(f"Accuracy: {avg_accuracy:.4f}, Configuration: {config}")

if __name__ == "__main__":
    main()
