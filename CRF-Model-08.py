# Algorithm 8 - CRF Model training
# Dataset cleaning, pre-processing XML and create slots and embeddings
# RNN Bidirectional LSTM Layer and CRF Layer
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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed

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

# Lists to store sentences and labels
sentences = []
labels = []

# Loop through files in directory
for file in files:
    if file.endswith(".xml"):
        output_html += f"<p>{file}</p>"
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        output_html += f"<h4>Conteúdo do arquivo {file}:</h4>"

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

                    # Apply word replacements
                    sentence_text = replace_words(sentence_text)

                    # Tokenize the sentences
                    sentences_list = sent_tokenize(sentence_text)

                    # Add labels
                    labels_list = []
                    for sent in sentences_list:
                        label = [1 if word == annotated_word else 0 for word in word_tokenize(sent)]
                        labels_list.append(label)

                    sentences.extend(sentences_list)
                    labels.extend(labels_list)

                    # Tokenize the sentences into words
                    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

                    # Obtain the maximum sequence length
                    max_sequence_length = max(len(sentence) for sentence in tokenized_sentences)

                    # Padding for the input sequences
                    X = pad_sequences(tokenized_sentences, maxlen=max_sequence_length, dtype='object')

                    # Padding for the label sequences
                    y = pad_sequences(labels, maxlen=max_sequence_length, dtype='object')

                    # Convert labels to categorical
                    y_categorical = tf.keras.utils.to_categorical(y)

                    # Certifique-se de que o número de amostras seja consistente
                    assert len(X) == len(y_categorical), "O número de amostras em X e y_categorical não é o mesmo."

                    # =============================================================================
                    # Construção do modelo RNN-CRF
                    model = Sequential()
                    model.add(Dense(units=100, activation='relu', input_dim=X.shape[1]))
                    model.add(Dropout(0.1))
                    model.add(Dense(units=100, activation='relu'))
                    model.add(Dropout(0.1))
                    model.add(Dense(units=max_sequence_length, activation='relu'))

                    # Learning rate
                    learning_rate = 0.01
                    rho = 0.9

                    # Optimizer
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho)

                    # Compilação do modelo
                    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                    # Definir paciência (patience) e EarlyStopping
                    patience = 10
                    early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

                    # Treinamento do modelo
                    model.fit(X, y_categorical, batch_size=32, epochs=60, callbacks=[early_stopping])

                    # Avaliação do modelo
                    loss, accuracy = model.evaluate(X, y_categorical)
                    print('Loss:', loss)
                    print('Accuracy:', accuracy)

                    # Predição usando o modelo treinado
                    predictions = model.predict(X)

                    # =============================================================================

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
                        output_html += f"<p>Rótulos {slot_number}: {y[sent_idx]}</p>"
                        output_html += "<pre>"

                        # Word Embeddings
                        for word in context_words:
                            word_embedding = tokenized_sent.wv[word].reshape((100, 1))
                            output_html += f"<p>{word}: {word_embedding}</p>"
                        output_html += "</pre>"

                        slot_number += 1

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
