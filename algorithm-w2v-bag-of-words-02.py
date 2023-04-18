import os
import re
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument

# Diretório da pasta com os arquivos de texto
dir_path = "c:/Dataset"

# Extrai todos os nomes de arquivos na pasta
file_names = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

# Inicializa uma lista para armazenar todos os textos dos arquivos
documents = []

# Percorre todos os arquivos na pasta
for file_name in file_names:
    # Lê o conteúdo do arquivo
    with open(os.path.join(dir_path, file_name), "r", encoding="utf-8") as f:
        text = f.read()

    # Limpa o texto, removendo caracteres indesejados e dividindo em tokens
    tokens = simple_preprocess(re.sub(r"[^a-zA-ZÀ-ú0-9\s]+", "", text.lower()))

    # Converte a lista de tokens em uma única string
    token_str = " ".join(tokens)

    # Adiciona o texto limpo à lista de documentos
    documents.append(TaggedDocument(token_str, [file_name]))

# Treina o modelo Word2Vec com Bag of Words
model = Word2Vec(documents, min_count=1, workers=4)

# Salva o modelo treinado em um arquivo
model.save("modelo.bin")
