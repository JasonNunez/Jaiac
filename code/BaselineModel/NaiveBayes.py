import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
import time
import re

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
    text = re.sub(' i m ', ' im ', text)  # Fixes i m -> im seperation
    text = re.sub(' e mail', ' email', text)  # Fixes e mail -> mail
    text = re.sub(' w e ', ' we ', text)  # Fixes w e -> we seperation
    text = re.sub(' we d ', ' wed ', text)  # Fixes we d -> wed seperation
    text = re.sub(' i d ', ' id ', text)  # Fixes i d -> id seperation
    text = re.sub(' they d ', ' theyd ', text)  # Fixes they d -> theyd seperation
    return text

df['text'] = df['text'].apply(clean_text)

df = df.drop_duplicates(subset='text')
df = df.dropna()

print("Text cleaned.")

X = df['text']
y = df['label']

# Define the K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=3270)

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
# Define the parameter grid
param_grid = {
    'vectorizer__max_features': [None, 500, 1000, 2000],
    'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'vectorizer__stop_words': [None, 'english'],
    'classifier__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'classifier__fit_prior': [True, False]
}

# Define the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1, verbose=4)

# Train the model with grid search
start_time = time.time()
grid_search.fit(X, y)
end_time = time.time()

# Print the best parameters and the corresponding accuracy
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_}")
print(f"Time taken for GridSearchCV: {end_time - start_time} seconds")


    
    