# Algorithm 6 - dataset cleaning, pre-processing XML and create slots and embeddings
# Results in file and browser

# Imports
import os
import xml.etree.ElementTree as ET
import webbrowser
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

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

# Tokenize the sentences into words
def tokenize_sentence(sentence):
    tokens = word_tokenize(sentence)
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords.words('portuguese') and token not in string.punctuation]
    return tokens

# Loop through files in directory
for file in files:
    if file.endswith(".xml"):
        output_html += f"<p>{file}</p>"
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        output_html += f"<h4>Conteúdo do arquivo {file}:</h4>"
        sentences = []

        # Loop throughsentences
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

                    # Prints the sentences and the annotated word
                    for sent_idx, sent in enumerate(sentences_list[:5]):  # Select up to 5 sentences
                        tokenized_sent = tokenize_sentence(sent)
                        annotated_index = tokenized_sent.index(
                            annotated_word.lower()) if annotated_word.lower() in tokenized_sent else -1
                        context_start = max(0, annotated_index - 5)
                        context_end = min(annotated_index + 6, len(tokenized_sent))
                        context_words = tokenized_sent[context_start:context_end]

                        context_text = ' '.join(context_words)
                        output_html += f"<p>Sentença {slot_number}: {context_text}</p>"
                        output_html += f"<p>Annotated Word: {annotated_word}</p>"
                        output_html += f"<p>Tokens: {context_words}</p>"  # Print the token vector
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
