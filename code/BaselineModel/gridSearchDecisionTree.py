import re

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import pandas as pd
import time

# Set option to display more rows
pd.set_option('display.max_rows', 500)

# Set option to display more columns
pd.set_option('display.max_columns', 50)

pd.set_option('display.max_colwidth', None)

# Load your dataset
df = pd.read_csv('text.csv')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()  # Removes all extra spaces
    text = re.sub(r'(href|http|img|src|span style).*', '', text)  # Removes all words to the right of urls
    text = re.sub(' t ', 't ', text)  # Fixes floating t's (don t -> dont)
    text = re.sub(' s ', 's ', text)  # Fixes floating s's (it s -> its)
    text = re.sub(' i m ', ' im ', text)  # Fixes i m -> im seperation
    text = re.sub(' e mail', ' email', text)  # Fixes e mail -> mail
    text = re.sub(' w e ', ' we ', text)  # Fixes w e -> we seperation
    text = re.sub(' we d ', ' wed ', text)  # Fixes we d -> wed seperation
    text = re.sub(' i d ', ' id ', text)  # Fixes i d -> id seperation
    text = re.sub(' they d ', ' theyd ', text)  # Fixes i d -> id seperation
    return text


df['text'] = df['text'].apply(clean_text)

df = df.drop_duplicates(subset='text')
df = df.dropna()

print("Text cleaned.")

# Define your text and labels
X = df['text']
y = df['label']

# Set up the pipeline
pipeline = Pipeline([
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

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=KFold(n_splits=5, shuffle=True, random_state=3270), verbose=3, n_jobs=-1)

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

