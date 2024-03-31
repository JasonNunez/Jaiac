import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline, Pipeline
import time
from prepare_data import load_and_clean_data

df = load_and_clean_data('text.csv')
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

X = df['text']
y = df['label']

# Define the K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=3270)


# Function to run the pipeline with a given configuration
def run_pipeline(config):
    # Create a new pipeline with the current configuration
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            binary=config['tfidf__binary'],
            max_features=config['tfidf__max_features'],
            ngram_range=config['tfidf__ngram_range'],
            stop_words=config['tfidf__stop_words'],
            use_idf=config['tfidf__use_idf']
        )),
        ('clf', LogisticRegression(
            max_iter=config['clf__max_iter']
        ))
    ])

    # Measure the time taken by the cross-validation process
    start_time = time.time()
    accuracies = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
    end_time = time.time()

    # Print the results
    print(f"Running configuration: {config}")
    print(f"Accuracies across folds: {accuracies}")
    print(f"Average accuracy: {np.mean(accuracies)}")
    print(f"Time taken: {end_time - start_time} seconds")
    print("---------------------------------------------------------")


# Iterate over the configurations and run the pipeline
for config_number, config_params in configurations.items():
    print(f"Running configuration number {config_number}")
    run_pipeline(config_params)
