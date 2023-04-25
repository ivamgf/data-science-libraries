# Algorithm cluster - 06
# dataset extract, create a bag of words, clustering with plot
# With Inertia table and Elbow Method

# imports
import os
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import re
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Define the list of stopwords for the Portuguese language
stop_words = set(stopwords.words('portuguese'))

# Create the Stemmer
stemmer = PorterStemmer()

# Path
path = "C:\Dataset"

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

# Tokenize the filtered sentences into words
words = [word_tokenize(sentence) for sentence in filtered_sentences]

# Build vocabulary
model = Word2Vec(words, min_count=1, workers=2)

# Train a Word2Vec model using the filtered sentences
model.train(words, total_examples=model.corpus_count, epochs=model.epochs)

# Get the word vector for each word
vectors = model.wv

# Define a range of values for k
k_values = range(1, 10)

# Initialize a list to store the inertia values for each k
inertia_values = []

# Apply K-means clustering
inertias = []
clusters_predictions = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(vectors.vectors)
    inertias.append(kmeans.inertia_)
    clusters_predictions.append(kmeans.fit_predict(vectors.vectors))
    print("Number of Clusters:", k, "\tInertia:", kmeans.inertia_)

# Plot the Elbow curve to determine the optimal number of clusters
fig, ax = plt.subplots()
ax.plot(range(1, 11), inertias)
ax.set_title('Elbow Method')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
plt.show()

# Create a table with the results
data = {'Number of Clusters': list(range(1, 11)), 'Inertia': inertias}
df = pd.DataFrame(data)
print("\nTable of Results:")
print(df)
