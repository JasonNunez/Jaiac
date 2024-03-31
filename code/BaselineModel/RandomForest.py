import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline
import re
import time

# Load the dataset
print("Loading data...")
df = pd.read_csv('text.csv')
print("Data loaded.")

# Preprocess the text
print("Preparing data...")


def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()  # Removes all extra spaces
    text = re.sub(r'(href|http|img|src|span style).*', '', text)  # Removes all words to the right of urls
    text = re.sub(' t ', 't ', text)  # Fixes floating t's (don t -> dont)
    text = re.sub(' s ', 's ', text)  # Fixes floating s's (it s -> its)
    text = re.sub(' i m ', ' im ', text)  # Fixes i m -> im separation
    text = re.sub(' e mail', ' email', text)  # Fixes e mail -> mail
    text = re.sub(' w e ', ' we ', text)  # Fixes w e -> we separation
    text = re.sub(' we d ', ' wed ', text)  # Fixes we d -> wed separation
    text = re.sub(' i d ', ' id ', text)  # Fixes i d -> id separation
    text = re.sub(' they d ', ' theyd ', text)  # Fixes they d -> theyd separation
    return text


df['text'] = df['text'].apply(clean_text)

df = df.drop_duplicates(subset='text')
df = df.dropna()

# Define your text and labels
print("Text cleaned.")

X = df['text']
y = df['label']

# Set up the pipeline
pipeline = Pipeline([
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