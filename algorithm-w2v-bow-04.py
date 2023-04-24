# Algorithm 1 - dataset cleaning and pre-processing

# Imports
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec

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
        content = nltk.re.sub('<.*?>', ' ', content)

        # Tokenize the content in the sentences
        sentences = sent_tokenize(content)

        # Flag for tokenizing and stemming the sentences after the keyword "ACORDAM"
        tokenize_after_keyword = False

        # List to store the filtered sentences
        filtered_sentences = []

        # Print sentences without stop words
        print("Sentences in the file " + file + " without stop words:")
        for sentence in sentences:
            if "RELATÓRIO" in sentence:
                tokenize_after_keyword = True

            if tokenize_after_keyword:
                # Tokenize each sentence into words
                words = word_tokenize(sentence.lower())

                # Remove stop words
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

                # Put the filtered words together into a sentence again
                filtered_sentence = ' '.join(words_stemming)

                # Add the filtered sentence to the list
                filtered_sentences.append(filtered_sentence)

# Train a Word2Vec model using the filtered sentences
tokenized_sentences = [sentence.split() for sentence in filtered_sentences]
model = Word2Vec(tokenized_sentences, min_count=1)

# Print the vocabulary
print("Vocabulary:")
print(list(model.wv.key_to_index.keys()))

# Get the word vector for a specific word
word = 'tutela'
if word in model.wv:
    print("Word vector for the word " + word + ":")
    print(model.wv[word])
else:
    print("Word '" + word + "' not in vocabulary.")

