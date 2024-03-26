import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
import re
import time

from sklearn.tree import DecisionTreeClassifier

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
    text = re.sub(' they d ', ' theyd ', text)  # Fixes i d -> id seperation
    return text


df['text'] = df['text'].apply(clean_text)

df = df.drop_duplicates(subset='text')
df = df.dropna()
y = df['label']

print("Text cleaned.")

# Define the K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=3270)

pipeline = make_pipeline(
    TfidfVectorizer(max_features=500,stop_words='english'),
    DecisionTreeClassifier(max_depth=10, random_state=3270)
)

# Measure the time taken by the K-Fold cross-validation process
start_time = time.time()
accuracies = cross_val_score(pipeline, df['text'], y, cv=kf, scoring='accuracy')
end_time = time.time()

# Print the results
print(f"Accuracies across folds: {accuracies}")
print(f"Average accuracy: {np.mean(accuracies)}")
print(f"Time taken for 5-Fold CV: {end_time - start_time} seconds")
