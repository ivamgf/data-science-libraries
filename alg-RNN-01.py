import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Exemplo de dados de treinamento
sentences = [
    "O requerente solicita o tratamento médico adequado para sua condição de saúde.",
    "A parte autora requer o fornecimento do medicamento prescrito pelo médico.",
    "É solicitado o acompanhamento médico constante para o requerente."
]
labels = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
]

# Construindo o modelo
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Dense(units=num_classes, activation='softmax'))

# Compilando o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Exemplo de texto para teste
text = "O requerente solicita o tratamento fisioterápico para sua recuperação."

# Pré-processamento do texto de teste
preprocessed_text = preprocess_text(text)

# Convertendo o texto pré-processado em sequência de tokens
input_sequence = text_to_sequence(preprocessed_text)

# Realizando a predição usando o modelo treinado
prediction = model.predict(input_sequence)

# Decodificando a predição para obter as etiquetas correspondentes
decoded_labels = decode_labels(prediction)

# Extraindo as palavras marcadas como "B-ReqTreatment"
treatment_words = extract_treatment_words(preprocessed_text, decoded_labels)

# Imprimindo as palavras marcadas
print("Palavras marcadas como 'B-ReqTreatment':")
for word in treatment_words:
    print(word)
