import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
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

print("Text cleaned.")

X = df['text']
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3270)

# Define the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=10, stop_words='english')

# Define the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=3270)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 10, 10, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Define the grid search with cross-validation
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# Measure the time taken by the grid search
start_time = time.time()
grid_search.fit(tfidf_vectorizer.fit_transform(X_train), y_train)
end_time = time.time()

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
print("Time taken for Grid Search: {:.2f} seconds".format(end_time - start_time))

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
accuracy = best_model.score(tfidf_vectorizer.transform(X_test), y_test)
print("Accuracy on Test Set:", accuracy)
