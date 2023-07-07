# Algorithm 5 - dataset cleaning, pre-processing XML and create slots and embeddings
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
                annotated_word = sentence.find(
                    ".//webanno.custom.Judgmentsentity/de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token").text.strip()
                annotated_word = replace_expression(annotated_word)  # Substitui o conteúdo de annotated_word

                sentence_text = ' '.join(tokens)

                # Verifica se a sentença já foi adicionada
                if sentence_text not in sentences:
                    sentences.append(sentence_text)

                    # Aplica as substituições de palavras
                    sentence_text = replace_words(sentence_text)

                    # Procura a expressão anotada na sentença e adiciona as tags <>
                    if annotated_word in sentence_text:
                        sentence_text = sentence_text.replace(annotated_word, "[annotation] " + annotated_word + "</>")

                    # Imprime a sentença e a palavra anotada
                    output_html += f"<p>Slot {slot_number}: {sentence_text}</p>"
                    output_html += f"<p>Annotated Word: {annotated_word}</p>"
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
