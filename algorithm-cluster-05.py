# Algorithm cluster - 05
# dataset extract, create a bag of words, clustering with plot and best_k

# imports
import os
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

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
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(vectors.vectors)
    inertias.append(kmeans.inertia_)

# Plot the Elbow curve to determine the optimal number of clusters
fig = px.line(x=list(range(1, 11)), y=inertias, title='Elbow Method')
fig.update_layout(xaxis_title='Number of Clusters', yaxis_title='Inertia')

# Apply K-means clustering with the optimal number of clusters
best_k = 3
kmeans = KMeans(n_clusters=best_k)
kmeans.fit(vectors.vectors)

# Print the clusters
print("Clusters:")
for i in range(best_k):
    cluster = []
    for j in range(len(vectors.vectors)):
        if kmeans.labels_[j] == i:
            cluster.append(vectors.index_to_key[j])
    print("Cluster ", i, ": ", cluster)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(vectors.vectors)

# Plot the clusters
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
fig, ax = plt.subplots()

for i in range(len(reduced)):
    ax.scatter(reduced[i, 0], reduced[i, 1], color=colors[kmeans.labels_[i]])
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:,0], centroids[:,1], marker='*', s=300, c='black')
plt.show()
