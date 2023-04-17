import os
import numpy as np
import gensim

# Define the path of the folder with the text files
path = "C:\\Dataset"

# Obter o caminho completo do arquivo do modelo word2vec
model_path = os.path.join(os.getcwd(), 'model.bin')

# Load the pre-trained word2vec model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# Function to load and process each text file
def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        words = text.split()
        vector_list = []
        for word in words:
            if word in w2v_model.vocab:
                vector_list.append(w2v_model[word])
        vector = np.mean(vector_list, axis=0)
        return vector

# Loop through all files in the specified folder and add the vectors to a list
vectors = []
for filename in os.listdir(path):
    if filename.endswith('.txt'):
        file_path = os.path.join(path, filename)
        vector = process_text_file(file_path)
        vectors.append(vector)

# Apply the "words to bag" method to generate a feature matrix
feature_matrix = np.vstack(vectors)
