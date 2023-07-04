# Algorithm 2 - dataset cleaning, pre-processing XML and create slots and embeddings
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

slot_number = 1

for file in files:
    if file.endswith(".xml"):
        output_html += f"<p>{file}</p>"
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        output_html += f"<h4>Conteúdo do arquivo {file}:</h4>"

        for sentence_element in root.iter("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"):
            pdf_chunk_found = False
            tokens = []

            for element in sentence_element.iter():
                if element.tag == "org.dkpro.core.api.pdf.type.PdfChunk" and not pdf_chunk_found:
                    pdf_chunk_found = True
                elif element.tag == "webanno.custom.Judgmentsentity" and pdf_chunk_found:
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

                        if len(tokens) > 0:
                            model = Word2Vec(min_count=1, workers=2, sg=1)
                            model.build_vocab([tokens])
                            model.train([tokens], total_examples=model.corpus_count, epochs=model.epochs)

                            output_html += "<h4>Vocabulary:</h4>"
                            for word in model.wv.key_to_index.keys():
                                output_html += f"<p>Slot: {slot_number}</p>"
                                output_html += "<p>Token:</p>"
                                output_html += f"<pre>{word}</pre>"
                                output_html += f"<p>Instance: {instance}</p>"
                                output_html += f"<p>Value: {value}</p>"
                                output_html += "<p>Word Embedding:</p>"
                                output_html += f"<pre>{model.wv[word]}</pre>"

                                # Create char embeddings for the token
                                char_embeddings = create_char_embeddings(word)

                                # Concatenate word and char embeddings
                                combined_embeddings = np.concatenate((model.wv[word], char_embeddings), axis=None)

                                # Print char embeddings
                                output_html += "<p>Char Embedding:</p>"
                                output_html += f"<pre>{char_embeddings}</pre>"

                                # Print the concatenated embeddings
                                output_html += "<p>Concatenated Embedding:</p>"
                                output_html += f"<pre>{combined_embeddings}</pre>"

                                slot_number += 1
                        else:
                            output_html += "<p>Nenhum token encontrado para treinar o modelo Word2Vec.</p>"

                    # Reset tokens for the next slot
                    tokens = []
                    pdf_chunk_found = False
                elif pdf_chunk_found:
                    if element.text:
                        token = element.text.strip().lower()
                        token = replace_words(token)
                        token = replace_expression(token)
                        tokens.append(token)

# Salva o resultado no arquivo de saída TXT
with open(output_file_txt, "w", encoding="utf-8") as f:
    f.write(output_html)

# Salva o resultado no arquivo de saída HTML
with open(output_file_html, "w", encoding="utf-8") as f:
    f.write(output_html)

# Abre o arquivo HTML no navegador
webbrowser.open(output_file_html)

print("Results saved in folder C://Outputs")
