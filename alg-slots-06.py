# Algorithm 6 - dataset cleaning, pre-processing XML and create slots and embeddings
# Results in file and browser

# Imports
import os
import xml.etree.ElementTree as ET
import webbrowser

path = "C:\\Dataset-TRT"
files = os.listdir(path)

output_dir = "C:\\Outputs"
os.makedirs(output_dir, exist_ok=True)

output_html = ""

output_html += "<h3>Arquivos encontrados no diretório:</h3>"

slot_number = 1

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

for file in files:
    if file.endswith(".xml"):
        output_html += f"<p>{file}</p>"
        xml_file = os.path.join(path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        output_html += f"<h4>Conteúdo do arquivo {file}:</h4>"
        sentences = []

        for sentence in root.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'):
            tokens = []
            for token in sentence.iter('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'):
                tokens.append(token.text.strip())

            if sentence.find(".//webanno.custom.Judgmentsentity") is not None:
                annotated_word = sentence.find(".//webanno.custom.Judgmentsentity/de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token").text.strip()
                annotated_word = replace_expression(annotated_word)
                sentence_text = ' '.join(tokens)

                if sentence_text not in sentences:
                    sentences.append(sentence_text)

                    sentence_text = replace_words(sentence_text)

                    if annotated_word in sentence_text:
                        start_index = sentence_text.find(annotated_word)
                        end_index = start_index + len(annotated_word)

                        # Procura até 5 palavras antes da expressão anotada
                        for i in range(1, 6):
                            if start_index - i >= 0 and sentence_text[start_index - i].isalnum():
                                start_index -= 1

                        # Procura até 5 palavras depois da expressão anotada
                        for i in range(1, 6):
                            if end_index + i < len(sentence_text) and sentence_text[end_index + i].isalnum():
                                end_index += 1

                        slot_part_before = sentence_text[start_index:end_index].strip()

                        # Verifica se há espaço suficiente para obter 5 palavras após a expressão anotada
                        if end_index + 5 < len(sentence_text):
                            slot_part_after = sentence_text[end_index:end_index + 5].strip()
                        else:
                            slot_part_after = sentence_text[end_index:].strip()

                        slot_part = slot_part_before + " " + slot_part_after

                        output_html += f"<p>Slot {slot_number}: {slot_part}</p>"
                        output_html += f"<p>Annotated Word: {annotated_word}</p>"
                        slot_number += 1

output_file_txt = os.path.join(output_dir, "output.txt")
output_file_html = os.path.join(output_dir, "output.html")

with open(output_file_txt, "w", encoding="utf-8") as f:
    f.write(output_html)

with open(output_file_html, "w", encoding="utf-8") as f:
    f.write(output_html)

webbrowser.open(output_file_html)

print("Results saved in folder C://Outputs")
