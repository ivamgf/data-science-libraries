# Algorithm 2 - dataset cleaning, pre-processing XML and create embeddings
# Implemented RNN with LSTMs
# Results in file and browser

# Imports
import os
import nltk
import xml.etree.ElementTree as ET
import spacy
from gensim.models import Word2Vec
from datetime import datetime
import hashlib
import webbrowser
import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
import numpy as np

nltk.download('punkt')

path = "C:\\Dataset-TRT"
files = os.listdir(path)

nlp = spacy.load("pt_core_news_sm")

# Function to replace words
def replace_words(text):
    word_replacements = {
        "LimitaÃ§Ã£o da condenaÃ§Ã£o aos valores dos pedidos": "Limitacao da condenacao aos valores dos pedidos",
        "AssÃ©dio moral": "Assedio moral",
        "HonorÃ¡rios sucumbenciais": "Honorarios sucumbenciais",
        "Estabilidade acidentÃ¡ria": "Estabilidade acidentaria",
        "DoenÃ§a ocupacional": "Doenca ocupacional"
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
        "DoenÃ§a ocupacional": "Doenca ocupacional"
    }
    for expression, replacement in expressions.items():
        text = text.replace(expression, replacement)
    return text

# Function to create char embeddings
def create_char_embeddings(word):
    char_indices = {char: i + 1 for i, char in enumerate(set(word))}
    text_indices = [char_indices[char] for char in word]
    return np.array(text_indices)

output_dir = "C:\\Outputs"
current_datetime = datetime.now()
timestamp = current_datetime.strftime("%Y%m%d%H%M%S")
hash_value = hashlib.md5(current_datetime.isoformat().encode()).hexdigest()

output_file_txt = os.path.join(output_dir, f"output_{timestamp}_{hash_value}.txt")
output_file_html = os.path.join(output_dir, f"output_{timestamp}_{hash_value}.html")

os.makedirs(output_dir, exist_ok=True)

output_html = ""

output_html += "<h3>Arquivos encontrados no diretório:</h3>"

for file in files:
    if file.endswith(".xml"):
        output_html += f"<p>{file}</p>"
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        output_html += f"<h4>Conteúdo do arquivo {file}:</h4>"
        sentences = []
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

                for child in element:
                    if child.text:
                        text = child.text.strip().lower()
                        text = replace_words(text)
                        text = replace_expression(text)
                        doc = nlp(text)
                        tokens = [token.text for token in doc]
                        sentences.append(tokens)

        if len(sentences) > 0:
            model = Word2Vec(sentences, min_count=1, workers=2, sg=1, window=5)
            model.build_vocab(sentences)
            model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

            output_html += "<h4>Vocabulary:</h4>"
            for word in model.wv.key_to_index.keys():
                output_html += f"<p>Token: {word}</p>"
                output_html += f"<p>Instance: {instance}</p>"
                output_html += f"<p>Value: {value}</p>"
                output_html += "<p>Word Embedding:</p>"
                output_html += f"<pre>{model.wv[word]}</pre>"

                # Create char embeddings for the token
                char_embeddings = create_char_embeddings(word)

                # Concatenate word and char embeddings
                combined = np.concatenate((model.wv[word], char_embeddings), axis=None)

                # Reshape the combined embeddings to match the expected input shape of the LSTM layer
                combined_embeddings = np.reshape(combined, (1, 1, -1))

                # Print char embeddings
                output_html += "<p>Char Embedding:</p>"
                output_html += f"<pre>{char_embeddings}</pre>"

                # Print the concatenated embeddings
                output_html += "<p>Concatenated Embedding:</p>"
                output_html += f"<pre>{combined_embeddings}</pre>"

                # LSTM model
                input_size = combined_embeddings.shape[-1]  # Updated input size
                hidden_size = 64
                num_classes = 10
                sequence_length = 1

                # Create LSTM model
                lstm_model = nn.Sequential(
                    nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True),
                    nn.Linear(hidden_size * 2, num_classes)
                )

                # Create CRF layer
                crf_layer = CRF(num_classes)

                # Apply CRF layer to the LSTM output
                def forward_pass(inputs):
                    lstm_output, _ = lstm_model(inputs)
                    crf_output = crf_layer(lstm_output)
                    return crf_output

                # Create model
                lstm_crf_model = forward_pass

                # Set device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                lstm_crf_model.to(device)

                # Define optimizer and loss function
                optimizer = optim.Adam(lstm_crf_model.parameters())
                loss_function = crf_layer.loss

                # Generate example data
                num_samples = 1
                X = torch.Tensor(combined_embeddings).to(device)
                y = torch.randint(num_classes, size=(num_samples, sequence_length)).to(device)

                # Train the model
                num_epochs = 10
                for epoch in range(num_epochs):
                    lstm_crf_model.train()
                    optimizer.zero_grad()

                    outputs = lstm_crf_model(X)
                    loss = loss_function(outputs, y)
                    loss.backward()
                    optimizer.step()

                # Print LSTM-CRF model results
                output_html += "<p>LSTM-CRF Model Results:</p>"
                lstm_crf_results = lstm_crf_model(X)
                output_html += f"<pre>{lstm_crf_results}</pre>"

        else:
            output_html += "<p>Nenhuma sentença encontrada para treinar o modelo Word2Vec.</p>"

# Save the results to output TXT file
with open(output_file_txt, "w", encoding="utf-8") as f:
    f.write(output_html)

# Save the results to output HTML file
with open(output_file_html, "w", encoding="utf-8") as f:
    f.write(output_html)

# Open the HTML file in the browser
webbrowser.open(output_file_html)

print("Results saved in folder C://Outputs")
