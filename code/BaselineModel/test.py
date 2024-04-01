import pandas as pd

# Load the CSV file
df = pd.read_csv('test.csv')

# Select the column you're interested in
# Replace 'your_column_name' with the actual column name you want to analyze
column_data = df['text']

# Initialize an empty set to store unique words
unique_words = set()

# Iterate over each cell in the column
for text in column_data:
    # Assuming each word is separated by a space, split the text into words
    # Convert to lowercase to count 'The' and 'the' as the same word, for example
    words = text.lower().split()

    # Update the set of unique words with words from this cell
    unique_words.update(words)

# The length of the set now represents the number of unique words
print("Number of unique words:", len(unique_words))
