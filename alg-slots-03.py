# Algorithm 3 - dataset cleaning, pre-processing XML and create slots and embeddings
# Results in file and browser

# Imports
import os
import xml.etree.ElementTree as ET
import webbrowser

# Caminho do diretório
path = "C:\\Dataset-TRT"
files = os.listdir(path)

output_dir = "C:\\Outputs"
os.makedirs(output_dir, exist_ok=True)

output_html = ""

output_html += "<h3>Arquivos encontrados no diretório:</h3>"

slot_number = 1

# Loop pelos arquivos no diretório
for file in files:
    if file.endswith(".xml"):
        output_html += f"<p>{file}</p>"
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        output_html += f"<h4>Conteúdo do arquivo {file}:</h4>"
        sentences = []

        # Loop pelas sentenças
        for sentence in root.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'):
            tokens = []
            for token in sentence.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'):
                tokens.append(token.text.strip())

            # Verifica se a sentença contém as tags específicas
            if sentence.find(".//webanno.custom.Judgmentsentity") is not None:
                sentence_text = ' '.join(tokens)
                sentences.append(sentence_text)

        # Imprime as sentenças encontradas
        for sentence in sentences:
            output_html += f"<p>Slot {slot_number}: {sentence}</p>"
            slot_number += 1

# Caminho dos arquivos de saída
output_file_txt = os.path.join(output_dir, "output.txt")
output_file_html = os.path.join(output_dir, "output.html")

# Salva o resultado no arquivo de saída TXT
with open(output_file_txt, "w", encoding="utf-8") as f:
    f.write(output_html)

# Salva o resultado no arquivo de saída HTML
with open(output_file_html, "w", encoding="utf-8") as f:
    f.write(output_html)

# Abre o arquivo HTML no navegador
webbrowser.open(output_file_html)

print("Results saved in folder C://Outputs")
