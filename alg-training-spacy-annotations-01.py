# Annotation training algorithm - 01

# Imports
import spacy
from spacy.lang.pt import Portuguese

# Create a new blank template
nlp = spacy.blank('pt')

# Add a New Custom Note Label
nlp.add_label('MINHA_ENTIDADE')

# Define the list of specific words
palavras_especificas = ['palavra1', 'palavra2', 'palavra3']

# Create the custom pipeline
def meu_pipe(doc):
    entidades = []
    for token in doc:
        if token.text in palavras_especificas:
            entidades.append((token.idx, token.idx + len(token.text), 'MINHA_ENTIDADE'))
    doc.ents = entidades
    return doc

# Add the Custom Pipeline to the Template
nlp.add_pipe(meu_pipe, last=True)

# Set the training data
dados_treinamento = {
    'texto1': [(0, 7, 'MINHA_ENTIDADE')],
    'texto2': [(8, 15, 'MINHA_ENTIDADE')],
    'texto3': [(16, 24, 'MINHA_ENTIDADE')]
}

# Start model training
n_iter = 10
for i in range(n_iter):
    perda = {}
    batches = spacy.util.minibatch(dados_treinamento, size=2)
    for batch in batches:
        texts = [nlp(text) for text in batch.keys()]
        annotations = batch.values()
        nlp.update(texts, annotations, losses=perda)
    print(perda)

# Test the trained model on a test dataset
dados_teste = {
    'texto4': [],
    'texto5': [],
    'texto6': []
}

for texto, anotacoes in dados_teste.items():
    doc = nlp
