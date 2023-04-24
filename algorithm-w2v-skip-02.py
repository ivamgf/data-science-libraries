# Algorithm w2v - Skip-gram - 01 - dataset extract and create a skip-gram

# Imports
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import re

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Define the list of stopwords for the Portuguese language
stop_words = set(stopwords.words('portuguese'))

# Create the Stemmer
stemmer = PorterStemmer()

# Use double backslashes in file or directory path
path = "C:\\Dataset"

# Get a list of files in the directory
files = os.listdir(path)

# Print the file list
print("Files found in the directory:")
for file in files:
    print(file)

    # Open the file and read the contents
    with open(os.path.join(path, file), 'r', encoding='utf8') as c:
        content = c.read()

        # Remove the text between the < and > symbols
        content = re.sub('<.*?>', ' ', content)

        # Tokenize the content in the sentences
        sentences = sent_tokenize(content)

        # List to store the filtered sentences
        filtered_sentences = []

        # Print sentences without stop words
        print("Sentences in the file " + file + " without stop words:")
        for sentence in sentences:
            # Remove the text between the < and > symbols
            sentence = re.sub('<.*?>', ' ', sentence)

            # Tokenize each sentence into words
            words = word_tokenize(sentence.lower())

            # Remove stop words and punctuation
            filtered_words = [
                word
                for word in words
                if word not in stop_words and
                   word not in punctuation
            ]

            # Apply stemming to each word
            words_stemming = [
                stemmer.stem(word)
                for word in filtered_words
            ]

            # Add the filtered sentence to the list
            filtered_sentences.append(words_stemming)

# Build vocabulary
model = Word2Vec(filtered_sentences, min_count=1, workers=2, sg=1, window=5)

# Train a Word2Vec model using the filtered sentences
model.train(filtered_sentences, total_examples=model.corpus_count, epochs=model.epochs)

# Print the vocabulary
print("Vocabulary:")
print(list(model.wv.key_to_index.keys()))

# Get the word vector for a specific word
for word in model.wv.key_to_index.keys():
    print("Word vector for the word => " + word + ":")
    print(model.wv[word])

