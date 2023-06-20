# Algorithm 9 - dataset cleaning, pre-processing XML and create embeddings

# Imports
import os
import nltk
import xml.etree.ElementTree as ET
import spacy
from gensim.models import Word2Vec

# Downloads
nltk.download('punkt')

# Use double backslashes in file or directory path
path = "C:\\Dataset-TRT"

# Get a list of files in the directory
files = os.listdir(path)

# Load SpaCy language model
nlp = spacy.load("pt_core_news_sm")

# Function to replace words in a text
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

# Function to replace expressions in the console output
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

# Print the file list
print("Arquivos encontrados no diretório:")
for file in files:
    if file.endswith(".xml"):
        print(file)

        # Open the XML file and parse the contents
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the XML tags and text content within the specific tags
        print("Conteúdo do arquivo " + file + ":")
        sentences = []  # List to store all sentences
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

                # Replace values
                instance = replace_words(instance)
                value = replace_words(value)

                # Iterate over the element's children and print their text content
                for child in element:
                    if child.text:
                        text = child.text.strip().lower()
                        text = replace_words(text)

                        # Replace expressions in the text
                        text = replace_expression(text)

                        # Tokenize the text using SpaCy
                        doc = nlp(text)
                        tokens = [token.text for token in doc]

                        sentences.append(tokens)  # Add the tokens to sentences list

                        model = Word2Vec(min_count=1, workers=2)
                        model.build_vocab(sentences)

                        # Train a Word2Vec model using the filtered sentences
                        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

                        # Print the tokens and attributes
                        tokensText = str(tokens)

                        # Print the vocabulary
                        print("Vocabulary:")
                        for word in model.wv.key_to_index.keys():
                            print("Token:" + word)
                            print("Instance:" + instance.encode("utf-8").decode("utf-8"))
                            print("Value:" + value.encode("utf-8").decode("utf-8"))
                            print("Embedding:")
                            print(model.wv[word])
