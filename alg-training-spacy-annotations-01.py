# Annotation training algorithm - 01

# Imports
import spacy
from spacy.lang.pt import Portuguese
import os

# Create a new blank template
nlp = spacy.blank('pt')

# Add a New Custom Note Label
labels = [
    'Doença ocupacional',
    'Acidente de trabalho',
    'Estabilidade acidentária',
    'Estabilidade da gestante',
    'Adicional de insalubridade',
    'Horas extras, validade do ponto e/ou da compensação',
    'Intervalos intrajornadas',
    'Horas in itinere',
    'Cargo de confiança bancário',
    'Assédio moral',
    'Desvio/acúmulo de funções',
    'Equiparação salarial',
    'Validade da dispensa por justa causa',
    'Desconsideração da personalidade jurídica',
    'Responsabilização da Adm. Pública',
    'Rescisão indireta',
    'Justiça gratuita',
    'Honorários sucumbenciais',
    'Limitação da condenação aos valores dos pedidos',
    'Juros e correção monetária',
    'Prescrição intercorrente'
]

for label in labels:
    nlp.add_label(label)
# nlp.add_label('MINHA_ENTIDADE')

# Define the list of specific words
palavras_especificas = ['palavra1', 'palavra2', 'palavra3']

# Create the custom pipeline
def meu_pipe(doc):
    entidades = []
    for token in doc:
        for label in labels:
            if label in token.text:
                entidades.append((token.idx, token.idx + len(token.text), label))
                doc.ents = entidades
        return doc

# Add the Custom Pipeline to the Template
nlp.add_pipe(meu_pipe, last=True)

# Set the training data
# dados_treinamento = {
#     'texto1': [(0, 7, 'MINHA_ENTIDADE')],
#     'texto2': [(8, 15, 'MINHA_ENTIDADE')],
#     'texto3': [(16, 24, 'MINHA_ENTIDADE')]
# }

dados_treinamento = []

path = "C:\Dataset"
for filename in os.listdir(path):
    if filename.endswith(".rtf"):
        with open(os.path.join(path, filename), 'r') as f:
            text = f.read()
            entities = []
    for label in labels:
        start = text.find(label)
    while start != -1:
        end = start + len(label)
        entities.append((start, end, label))
        start = text.find(label, end)
        dados_treinamento.append((text, {'entities': entities}))

# Start model training
n_iter = 10
for i in range(n_iter):
    perda = {}
    batches = spacy.util.minibatch(dados_treinamento, size=2)
    for batch in batches:
        texts = [nlp(text) for text, annotations in batch]
        annotations = [annotations for text, annotations in batch]
        nlp.update(texts, annotations, losses=perda)
    print(perda)

# Test the trained model on a test dataset
dados_teste = {
    'texto1': 'Um funcionário sofreu um acidente de trabalho na empresa',
    'texto2': 'A gestante tem direito à estabilidade no emprego',
    'texto3': 'O empregado tem direito a receber adicional de insalubridade'
}

# Numbers of array to be a positions of characters in text
# Adicionar anotações para cada texto de teste
dados_teste['texto1'] = [(26, 43, 'Acidente de trabalho')]
dados_teste['texto2'] = [(21, 31, 'Estabilidade da gestante')]
dados_teste['texto3'] = [(36, 57, 'Adicional de insalubridade')]

for texto, anotacoes in dados_teste.items():
    doc = nlp(texto)
    print(texto, [(ent.text, ent.label_) for ent in doc.ents])