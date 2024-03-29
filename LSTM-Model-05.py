# Algorithm 5 - LSTM Model training
# Dataset cleaning, pre-processing XML and create slots and embeddings
# RNN Bidiretional LSTM Layer with cross-validation
# Results in file and browser

# Imports
import os
import xml.etree.ElementTree as ET
import webbrowser
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from keras.layers import Dense

# Downloads
nltk.download('punkt')

# Directory path
path = "C:\\Dataset-TRT"
files = os.listdir(path)

output_dir = "C:\\Outputs"
os.makedirs(output_dir, exist_ok=True)

output_html = ""

output_html += "<h3>Arquivos encontrados no diretório:</h3>"

slot_number = 1

# Functions

# Function to replace words
def replace_words(text):
    word_replacements = {
        "LimitaÃ§Ã£o da condenaÃ§Ã£o aos valores dos pedidos": "Limitacao da condenacao aos valores dos pedidos",
        "AssÃ©dio moral": "Assedio moral",
        "HonorÃ¡rios sucumbenciais": "Honorarios sucumbenciais",
        "Estabilidade acidentÃ¡ria": "Estabilidade acidentaria",
        "DoenÃ§a ocupacional": "Doenca ocupacional",
        "doenÃ§a": "doenca",
        "doenÃ§as": "doencas",
        "rÃ©": "re",
        "nÃ£o": "nao",
        "sÃ£o": "sao",
        "forÃ§a": "foram",
        "benefÃ­cio": "beneficio",
        "auxÃ­lio": "auxilio",
        "previdenciÃ¡rio": "previdenciario",
        "existÃªncia": "existencia",
        "necessÃ¡rio": "necessario",
        "contrÃ¡rio": "contrario",
        "vÃ¡lvula": "valvula"
    }
    for old_word, new_word in word_replacements.items():
        text = text.replace(old_word, new_word)
    return text

# Function to replace expressions
def replace_expression(text):
    expressions = {
        "LimitaÃ§Ã£o da condenaÃ§Ã£o aos valores dos pedidos": "Limitacao da condenacao aos valores dos pedidos",
        "AssÃ©dio moral": "Assedio moral",
        "HonorÃ¡rios sucumbenciais": "Honorarios sucumbenciais",
        "Estabilidade acidentÃ¡ria": "Estabilidade acidentaria",
        "DoenÃ§a ocupacional": "Doenca ocupacional",
        "doenÃ§a": "doenca",
        "doenÃ§as": "doencas"
    }
    for expression, replacement in expressions.items():
        text = text.replace(expression, replacement)
    return text

# Tokenize the sentences into words and create skipgram Word2Vec
def tokenize_sentence(sentence):
    tokens = word_tokenize(sentence)
    tokens = [token.lower() for token in tokens if token.lower() not in string.punctuation]

    # Create skipgram Word2Vec model for the sentence
    model = Word2Vec(sentences=[tokens], min_count=1, workers=2, sg=1, window=5)

    return model

# Function to split the data into 5 parts
def split_data(sentences, num_folds=5):
    fold_size = len(sentences) // num_folds
    data_folds = []
    for i in range(num_folds):
        fold_start = i * fold_size
        fold_end = fold_start + fold_size
        data_folds.append(sentences[fold_start:fold_end])
    return data_folds

# Function to get the training and testing data for a specific fold
def get_fold_data(data_folds, fold_idx):
    training_data = []
    testing_data = []
    for i, data_fold in enumerate(data_folds):
        if i == fold_idx:

            testing_data.extend(data_fold)
        else:
            training_data.extend(data_fold)
    return training_data, testing_data

# List to store the results for each fold
fold_results = []

# Loop through files in directory
for file in files:
    if file.endswith(".xml"):
        output_html += f"<p>{file}</p>"
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        output_html += f"<h4>Conteúdo do arquivo {file}:</h4>"
        sentences = []

        # Loop through sentences
        for sentence in root.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'):
            tokens = []
            for token in sentence.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'):
                tokens.append(token.text.strip())

            # Checks if the sentence contains the specific tags
            if sentence.find(".//webanno.custom.Judgmentsentity") is not None:
                annotated_word = sentence.find(
                    ".//webanno.custom.Judgmentsentity/de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token").text.strip()
                annotated_word = replace_expression(annotated_word)  # Substitui o conteúdo de annotated_word

                sentence_text = ' '.join(tokens)

                # Checks if the sentence has already been added
                if sentence_text not in sentences:
                    sentences.append(sentence_text)

    # Split data into 5 parts (5-fold cross-validation)
    data_folds = split_data(sentences)

    # Perform 5-fold cross-validation
    for fold_idx in range(5):
        slot_number = 1

        # Get training and testing data for this fold
        training_data, testing_data = get_fold_data(data_folds, fold_idx)

        # Tokenize and create the Word2Vec model only if training data is available
        if training_data:
            all_tokens = [word_tokenize(sentence) for sentence in training_data]
            all_tokens = [token.lower() for tokens in all_tokens for token in tokens if
                          token.lower() not in string.punctuation]
            model = Word2Vec(sentences=all_tokens, min_count=1, workers=2, sg=1, window=5)

           # Initialize a list to store results for this fold
            fold_results = []

# Loop through files in directory
for file in files:
    if file.endswith(".xml"):
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        sentences = []

        # Loop through sentences
        for sentence in root.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'):
            tokens = []
            for token in sentence.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'):
                tokens.append(token.text.strip())

            # Checks if the sentence contains the specific tags
            if sentence.find(".//webanno.custom.Judgmentsentity") is not None:
                annotated_word = sentence.find(
                    ".//webanno.custom.Judgmentsentity/de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token").text.strip()
                annotated_word = replace_expression(
                    annotated_word)  # Substitui o conteúdo de annotated_word

                sentence_text = ' '.join(tokens)

                # Checks if the sentence has already been added
                if sentence_text not in sentences:
                    sentences.append(sentence_text)

            # Loop through sentences again to process them for LSTM model (separated from the previous loop)
            for sentence in root.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'):
                tokens = []
                for token in sentence.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'):
                    tokens.append(token.text.strip())

                # Checks if the sentence contains the specific tags
                if sentence.find(".//webanno.custom.Judgmentsentity") is not None:
                    annotated_word = sentence.find(
                        ".//webanno.custom.Judgmentsentity/de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token").text.strip()
                    annotated_word = replace_expression(annotated_word)  # Substitui o conteúdo de annotated_word

                    sentence_text = ' '.join(tokens)

                    # Checks if the sentence has already been added
                    if sentence_text not in sentences:
                        sentences.append(sentence_text)

                        # Collect all sentences from the loop
                        all_sentences = sentences

                        # Apply word replacements
                        sentence_text = replace_words(sentence_text)

                        # Tokenize the sentences
                        sentences_list = sent_tokenize(sentence_text)

                        # Prints the sentences and the annotated word
                        for sent_idx, sent in enumerate(sentences_list[:5]):  # Select up to 5 sentences
                            tokenized_sent = tokenize_sentence(sent)
                            annotated_index = tokenized_sent.wv.key_to_index.get(
                                annotated_word.lower(), -1)
                            context_start = max(0, annotated_index - 5)
                            context_end = min(annotated_index + 6, len(tokenized_sent.wv.key_to_index))
                            context_words = list(tokenized_sent.wv.key_to_index.keys())[context_start:context_end]
                            context_words.reverse()  # Reverse the word order

                        # Print the Instance and Value attributes
                        for element in root.iter("webanno.custom.Judgmentsentity"):
                            if (
                                    "sofa" in element.attrib and
                                    "begin" in element.attrib and
                                    "end" in element.attrib and
                                    "Instance" in element.attrib and
                                    "Value" in element.attrib
                            ):
                                sofa = element.attrib["sofa"]
                                begin = element.attrib["begin"]
                                end = element.attrib["end"]
                                instance = element.attrib["Instance"]
                                value = element.attrib["Value"]

                                instance = replace_words(instance)
                                value = replace_words(value)

                        context_text = ' '.join(context_words)
                        context_text = context_text.replace(annotated_word, f"[annotation]{annotated_word}[annotation]")
                        output_html += f"<p>Sentença {slot_number}: {context_text}</p>"
                        output_html += f"<p>Annotated Word: {annotated_word}</p>"
                        output_html += f"<p>Instance: {instance}</p>"
                        output_html += f"<p>Value: {value}</p>"

                        # Print the token vector
                        output_html += f"<p>Slot de Tokens {slot_number}: {context_words}</p>"
                        output_html += "<pre>"

                        # Word Embeddings
                        for word in context_words:
                            word_embedding = tokenized_sent.wv[word].reshape((100, 1))
                            output_html += f"<p>{word}: {word_embedding}</p>"
                        output_html += "</pre>"

                        # Bidirectional LSTM model
                        input_size = word_embedding.shape[-1]
                        hidden_size = 64
                        num_classes = 10
                        sequence_length = 1

                        # Transpose input
                        word_embedding = np.transpose(word_embedding, (1, 0))

                        # Generate example data
                        num_samples = 1
                        # Reshape the input data
                        X = word_embedding.reshape((num_samples, 1, 100))
                        y = tf.random.uniform((num_samples, num_classes))

                        # Create Bidirectional LSTM model
                        lstm_model = tf.keras.Sequential()
                        lstm_model.add(Dense(units=32))
                        lstm_model.add(
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                                hidden_size, input_shape=(1, 120), dropout=0.1)))
                        lstm_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

                        # Learning rate
                        learning_rate = 0.01
                        rho = 0.9

                        # Optimizer
                        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho)

                        # Compile the model
                        lstm_model.compile(
                            loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

                        # Define patience and EarlyStopping
                        patience = 10
                        early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

                        # Train the model with EarlyStopping
                        lstm_model.fit(X, y, epochs=60, batch_size=32, callbacks=[early_stopping])

                        # Print LSTM model results
                        output_html += "<p>Bidirectional LSTM Model Results:</p>"
                        lstm_results = lstm_model.predict(X)
                        output_html += "<pre>"
                        # Get the indices of the words in the Slot de Tokens
                        word_indices = [tokenized_sent.wv.key_to_index[word.lower()] for word in context_words]
                        # Create a dictionary mapping word indices to LSTM results
                        results_dict = dict(zip(word_indices, lstm_results[0]))
                        # Iterate over the words in the Slot de Tokens and print the corresponding LSTM result
                        for word in context_words:
                            word_index = tokenized_sent.wv.key_to_index[word.lower()]
                            result = results_dict.get(word_index, 0.0)  # Default to 0.0 if word index not found
                            if word == annotated_word:
                                result = results_dict.get(word_index, 0.0)
                            output_html += f"<p>{word}: {result}"
                            if word != annotated_word:
                                output_html += " - [O]"
                            if word == annotated_word:
                                output_html += " - [B-ReqTreatment]"
                            output_html += "</p>"
                        output_html += "</pre>"

                        slot_number += 1

                        # Append the results for this fold to the list
                        fold_results.append(lstm_results)

                        # Calculate and print average results for all folds
                        average_results = np.mean(fold_results, axis=0)
                        output_html += f"<p>Average LSTM Model Results: {average_results}</p>"

# Output files path
output_file_txt = os.path.join(output_dir, "output.txt")
output_file_html = os.path.join(output_dir, "output.html")

# Save the result to the output TXT file
with open(output_file_txt, "w", encoding="utf-8") as f:
    f.write(output_html)

# Save the result to the HTML output file
with open(output_file_html, "w", encoding="utf-8") as f:
    f.write(output_html)

# Opens the HTML file in the browser
webbrowser.open(output_file_html)

print("Results saved in folder C://Outputs")
