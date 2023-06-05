import os
import subprocess

def convert_xml_to_rtf(xml_file, rtf_file):
    # Comando para executar o pandoc e converter XML para RTF
    command = f"pandoc -s {xml_file} -o {rtf_file}"

    # Executar o comando usando o subprocesso
    subprocess.run(command, shell=True)

# Diretório dos arquivos XML
path = "C:\\Dataset-TRT"

# Percorrer todos os arquivos no diretório
for filename in os.listdir(path):
    if filename.endswith(".xml"):
        xml_file = os.path.join(path, filename)
        rtf_file = os.path.join(path, f"{os.path.splitext(filename)[0]}.rtf")
        convert_xml_to_rtf(xml_file, rtf_file)
