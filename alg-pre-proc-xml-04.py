# Algorithm 4 - dataset cleaning and pre-processing XML

# Imports
import os
import nltk
import json
import re
import xml.etree.ElementTree as ET

# Downloads
nltk.download('punkt')

# Use double backslashes in file or directory path
path = "C:\\Dataset-TRT"
imports_path = "C:\\Imports\\instances.json"

# Get a list of files in the directory
files = os.listdir(path)

# Load word replacements from JSON
with open(imports_path, "r", encoding="utf-8") as json_file:
    word_replacements = json.load(json_file)

# Function to replace words in a text
def replace_words(text):
    for old_word, new_word in word_replacements.items():
        if isinstance(new_word, list):
            if text in new_word:
                text = new_word[new_word.index(text)]
        else:
            text = new_word if text == old_word else text
    return text

# Function to replace expressions in the console output
def replace_expression(text):
    expressions = {
        "LimitaÃ§Ã£o da condenaÃ§Ã£o aos valores dos pedidos": "Limitacao da condenacao aos valores dos pedidos",
        "AssÃ©dio moral": "Assedio moral"
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

                # Replace values from JSON
                instance = replace_words(instance)
                value = replace_words(value)

                # Iterate over the element's children and print their text content
                for child in element:
                    if child.text:
                        text = child.text.strip().lower()
                        text = replace_words(text)

                        # Replace expressions in the text
                        text = replace_expression(text)

                        # Print the replaced text and attributes
                        print(
                            text.encode("utf-8").decode("utf-8") +
                            "<" + instance.encode("utf-8").decode("utf-8") +
                            ">" + "<" + value.encode("utf-8").decode("utf-8") + ">"
                        )
