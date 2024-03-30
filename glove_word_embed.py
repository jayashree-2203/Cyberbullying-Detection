import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from nltk import download
import time
import csv
import os

def word_embedded_technique():
    # Download the Punkt tokenizer models if not already downloaded
    download('punkt')

    # Assuming 'preprocessed_cyberbullyingtweets.csv' has a column named 'preprocessed_text'
    df = pd.read_csv('./output/Preprocessed/preprocessed_data.csv')
    file_path = './output/Glove_word_embedded/glove_results.csv'

    # Handle non-string values in 'preprocessed_text'
    df['preprocessed_text'] = df['preprocessed_text'].astype(str)

    # Tokenize the text
    tokenized_text = df['preprocessed_text'].apply(lambda x: word_tokenize(x) if isinstance(x, str) else [])

    # Remove empty lists (resulting from non-string values)
    tokenized_text = tokenized_text[tokenized_text.apply(lambda x: len(x) > 0)]

    # Load the pre-trained GloVe model
    glove_model_file = 'glove_twitter_model.txt'
    word2vec_output_file = 'glove_model_word2vec_format.txt'

    # Convert GloVe model to Word2Vec format
    glove_model = KeyedVectors.load_word2vec_format(glove_model_file, binary=False, no_header=True)
    glove_model.save_word2vec_format(word2vec_output_file, binary=False)

    # Load the converted Word2Vec format model
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_output_file)

    # Read words from 'bad.txt' file
    with open('bullywords.txt', 'r') as file:
        words_to_check = [line.strip() for line in file]

    # Check if each word is in the vocabulary before getting its vector
    vectors_for_words = {word: word2vec_model[word] if word in word2vec_model else None for word in words_to_check}

    # Example: Find similar words for each word
    similar_words_for_each = {word: [item[0] for item in word2vec_model.most_similar(word, topn=5)] if word in word2vec_model else None for word in words_to_check}

    # Create a DataFrame with the vectors and similar words
    results_df = pd.DataFrame({
        'Word': words_to_check,
        'Vector': [vectors_for_words[word] for word in words_to_check],
        'Similar_Words': [similar_words_for_each[word] for word in words_to_check]
    })

    # Save the DataFrame to a new CSV file
    results_df.to_csv(file_path, index=False)
    timer();
    
    # Print vectors and similar words
    for word, vector in vectors_for_words.items():
        print(f"Vector for '{word}': {vector}")

    for word, similar_words in similar_words_for_each.items():
        print(f"Similar words to '{word}': {similar_words}")

    print('\n')
    print(results_df)
    count_rows(file_path);
    num_rows = count_rows(file_path)
    print(f"\n\nThe total number of rows in {file_path} is: {num_rows-1}")

    print("\n It performed matrix word embedding which transforms textual information into an array of real valued vectors. Once the GloVe has been trained, each word is assigned to a unique real-valued vector.\n\nThe output file is saved at:", file_path)


def count_rows(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        row_count = len(list(reader))
    return row_count

def timer():
    for progress in range(10, 101, 10):
        # Print the progress percentage
        print(f"Processing: {progress}%")

        # Simulate processing time using time.sleep()
        time.sleep(0.2)  # Adjust the sleep duration as needed
