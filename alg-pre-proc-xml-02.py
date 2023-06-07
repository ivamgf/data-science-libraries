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

        # Extract content from tags with the "Instance" attribute
        extracted_content = []
        for tag in root.findall(".//*[@Instance]"):
            extracted_content.append(tag.text)

        # Print the extracted content
        print("Conteúdo do arquivo", filename)
        for content in extracted_content:
            print(content)
