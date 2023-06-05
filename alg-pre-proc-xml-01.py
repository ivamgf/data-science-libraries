# Algorithm 1 - dataset cleaning and pre-processing XML

# Imports
import os
import xml.etree.ElementTree as ET

# Directory path
path = "C:\\Dataset-TRT"

# Iterate over the files in the directory
for filename in os.listdir(path):
    if filename.endswith(".xml"):
        # Create the file path
        filepath = os.path.join(path, filename)

        # Parse the XML file
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Print the extracted data
        print("Conteúdo do arquivo", filename)
        print(ET.tostring(root, encoding='utf-8').decode('utf-8'))
        print()  # Adiciona uma linha em branco para separar o conteúdo dos arquivos

