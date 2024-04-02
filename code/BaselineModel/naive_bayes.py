#!/usr/bin/env python3
"""
Multinomial Naïve Bayes model for emotional text classification.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from prepare_data import load_and_clean_data

__author__ = 'Alec Neal'
__version__ = 'Spring 2024'
__pylint__ = '2.14.5'


def run_pipeline(configuration, data, labels, c_val):
    """
    Runs the pipeline for a given configuration on the provided dataset.
    Returns the average accuracy and configuration details for further processing.

    Args:
        configuration (dict): Configuration settings for the CountVectorizer
        and MultinomialNB classifier.
        data (pandas.Series): The text data to classify.
        labels (pandas.Series): The labels for the text data.
        cross_validator (KFold): The cross-validation splitting strategy.
    """
    pline = Pipeline([
        ('vectorizer', CountVectorizer(
            max_features=configuration['vectorizer__max_features'],
            ngram_range=configuration['vectorizer__ngram_range'],
            stop_words=configuration['vectorizer__stop_words']
        )),
        ('classifier', MultinomialNB(
            alpha=configuration['classifier__alpha'],
            fit_prior=configuration['classifier__fit_prior']
        ))
    ])

    accuracies = cross_val_score(pline, data, labels, cv=c_val, scoring='accuracy', n_jobs=-1)

    avg_accuracy = np.mean(accuracies)
    return avg_accuracy, configuration


def main():
    """
    Main function to load data, define cross-validator, and run configurations.
    """
    dataframe = load_and_clean_data('dev.csv')
    configurations = {
        1: {
            'classifier__alpha': 0.1,
            'classifier__fit_prior': True,
            'vectorizer__max_features': 2000,
            'vectorizer__ngram_range': (1, 2),
            'vectorizer__stop_words': 'english'
        },
        2: {
            'classifier__alpha': 0.01,
            'classifier__fit_prior': True,
            'vectorizer__max_features': 2000,
            'vectorizer__ngram_range': (1, 2),
            'vectorizer__stop_words': 'english'
        },
        3: {
            'classifier__alpha': 1.0,
            'classifier__fit_prior': True,
            'vectorizer__max_features': 2000,
            'vectorizer__ngram_range': (1, 2),
            'vectorizer__stop_words': 'english'
        },
        4: {
            'classifier__alpha': 0.1,
            'classifier__fit_prior': True,
            'vectorizer__max_features': 2000,
            'vectorizer__ngram_range': (1, 3),
            'vectorizer__stop_words': 'english'
        },
        5: {
            'classifier__alpha': 0.01,
            'classifier__fit_prior': True,
            'vectorizer__max_features': 2000,
            'vectorizer__ngram_range': (1, 3),
            'vectorizer__stop_words': 'english'
        }
    }

    x_text = dataframe['text']
    y_labels = dataframe['label']

    kfold = KFold(n_splits=5, shuffle=True, random_state=3270)

    results = []

    for config_number, config_params in configurations.items():
        print(f"Running configuration number {config_number}")
        avg_accuracy, config = run_pipeline(config_params, x_text, y_labels, kfold)
        results.append((avg_accuracy, config))

    # Sort the results by average accuracy in descending order
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)

    # Print summary of top configurations
    print("Top Multinomial Naive Bayes Configurations:")
    for avg_accuracy, config in sorted_results:
        print(f"Accuracy: {avg_accuracy:.4f}, Configuration: {config}")


if __name__ == '__main__':
    main()
