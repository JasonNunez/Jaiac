import pandas as pd
import re


def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()  # Removes all extra spaces
    text = re.sub(r'(href|http|img|src|span style).*', '', text)  # Removes all words to the right of urls
    text = re.sub(' t ', 't ', text)  # Fixes floating t's (don t -> dont)
    text = re.sub(' s ', 's ', text)  # Fixes floating s's (it s -> its)
    text = re.sub(' i m ', ' im ', text)  # Fixes i m -> im separation
    text = re.sub(' e mail', ' email', text)  # Fixes e mail -> email
    text = re.sub(' w e ', ' we ', text)  # Fixes w e -> we separation
    text = re.sub(' we d ', ' wed ', text)  # Fixes we d -> wed separation
    text = re.sub(' i d ', ' id ', text)  # Fixes i d -> id separation
    text = re.sub(' they d ', ' theyd ', text)  # Fixes they d -> theyd separation
    return text


def load_and_clean_data(file_path):
    """
    Load the dataset and apply cleaning and preprocessing steps.
    """
    print("Loading data...")
    df = pd.read_csv(file_path)
    print("Data loaded.")

    print("Preparing data...")
    df['text'] = df['text'].apply(clean_text)
    df = df.drop_duplicates(subset='text')
    df = df.dropna()
    print("Text cleaned.")

    return df
