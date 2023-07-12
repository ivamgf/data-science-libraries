# RNN CRF Model Basic

# Imports
import numpy as np
import tensorflow as tf
from keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from gensim.models import Word2Vec

# Sentenças de treinamento e seus rótulos correspondentes
sentences = [
    ['Esta', 'é', 'uma', 'frase'],
    ['Python', 'é', 'uma', 'linguagem', 'de', 'programação'],
    ['Python', 'é', 'amigável'],
    ['Eu', 'gosto', 'de', 'Python'],
    ['Programar', 'em', 'Python', 'é', 'divertido']
]
labels = [
    [0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0, 0]
]

# Criação de um modelo Word2Vec usando as sentenças
model_w2v = Word2Vec(sentences, min_count=1)

# Obtém o tamanho máximo da sequência de palavras
max_sequence_length = max(len(sentence) for sentence in sentences)

# Transforma as palavras em vetores usando o Word2Vec e calcula a média dos vetores de palavras de cada sentença
X = np.array([
    np.mean([model_w2v.wv[word] for word in sentence], axis=0)
    for sentence in sentences
])

# Padding para as sequências de entrada
X = pad_sequences(X, maxlen=max_sequence_length, dtype='float32')

# Padding para as sequências de rótulos
y = pad_sequences(labels, maxlen=max_sequence_length, value=0, padding='post')

# Validação cruzada de 5 partes
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Construção do modelo RNN-CRF
    model = Sequential()
    model.add(Dense(units=100, activation='relu', input_dim=X.shape[1]))
    model.add(Dropout(0.1))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=max_sequence_length, activation='relu'))

    # Learning rate
    learning_rate = 0.01
    rho = 0.9

    # Optimizer
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho)

    # Compilação do modelo
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Definir paciência (patience) e EarlyStopping
    patience = 10
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

    # Treinamento do modelo
    model.fit(X_train, np.argmax(y_train, axis=-1), batch_size=32, epochs=60, callbacks=[early_stopping])

    # Avaliação do modelo
    loss, accuracy = model.evaluate(X_test, np.argmax(y_test, axis=-1))
    print('Loss:', loss)
    print('Accuracy:', accuracy)

    # Predição usando o modelo treinado
    predictions = model.predict(X_test)
