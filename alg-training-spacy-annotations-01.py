import spacy
from spacy.lang.pt import Portuguese

# Criar um novo modelo em branco
nlp = spacy.blank('pt')

# Adicionar um novo rótulo de anotação personalizado
nlp.add_label('MINHA_ENTIDADE')

# Definir a lista de palavras específicas
palavras_especificas = ['palavra1', 'palavra2', 'palavra3']

# Criar o pipeline personalizado
def meu_pipe(doc):
    entidades = []
    for token in doc:
        if token.text in palavras_especificas:
            entidades.append((token.idx, token.idx + len(token.text), 'MINHA_ENTIDADE'))
    doc.ents = entidades
    return doc

# Adicionar o pipeline personalizado ao modelo
nlp.add_pipe(meu_pipe, last=True)

# Definir os dados de treinamento
dados_treinamento = {
    'texto1': [(0, 7, 'MINHA_ENTIDADE')],
    'texto2': [(8, 15, 'MINHA_ENTIDADE')],
    'texto3': [(16, 24, 'MINHA_ENTIDADE')]
}

# Iniciar o treinamento do modelo
n_iter = 10
for i in range(n_iter):
    perda = {}
    batches = spacy.util.minibatch(dados_treinamento, size=2)
    for batch in batches:
        texts = [nlp(text) for text in batch.keys()]
        annotations = batch.values()
        nlp.update(texts, annotations, losses=perda)
    print(perda)

# Testar o modelo treinado em um conjunto de dados de teste
dados_teste = {
    'texto4': [],
    'texto5': [],
    'texto6': []
}

for texto, anotacoes in dados_teste.items():
    doc = nlp
