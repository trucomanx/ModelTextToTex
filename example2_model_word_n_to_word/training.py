#!/usr/bin/python3
#https://lejdiprifti.com/2023/10/14/creating-a-text-generation-neural-network-with-tensorflow/

################################################################################
from tensorflow.keras.preprocessing.text import tokenizer_from_json

with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()

tokenizer = tokenizer_from_json(tokenizer_json)
total_words = len(tokenizer.word_index) + 1

################################################################################
import tensorflow as tf
import json

dataset = tf.data.Dataset.load("meu_dataset")

for sample in dataset.take(1):
    print(sample[0])
    print(sample[1])

input_shape = sample[0].shape[1]

with open('input_shape.json', 'w') as archivo_json:
    json.dump({"input_shape":input_shape}, archivo_json)

################################################################################
import mymodules.model as mmm

model_1 = mmm.get_model(total_words, input_shape)


# sparse_categorical_crossentropy. Isso permite que o modelo use um número inteiro como rótulo, sem precisar converter para one-hot encoding.
model_1.compile(loss=tf.losses.SparseCategoricalCrossentropy(), 
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                metrics=['accuracy'])

model_1.summary()

################################################
'''
Como o modelo "entende" que são sequências?

1️⃣ Formato dos dados de entrada

    Seu xs tem dimensão (N, 107), ou seja, cada amostra tem 107 números (índices de palavras).
    Como passamos todas as palavras de uma vez para o modelo, ele vê cada amostra como um grupo de palavras relacionadas.

2️⃣ A camada Embedding

    Converte cada número (índice de palavra) em um vetor denso de tamanho 128.
    Resultado: A entrada, que antes era (107,), vira um tensor (107, 128).
    Agora, temos uma sequência de vetores e não só números.

3️⃣ A camada LSTM

    O LSTM lê os vetores da sequência um por um, na ordem, e processa a relação entre eles.
    Como a camada mantém um estado interno (memória) ao longo do tempo, ela entende a sequência como algo conectado, e não apenas como números soltos.

4️⃣ A saída da segunda LSTM (32 neurônios)

    Retorna um único vetor de tamanho 32, que contém a "memória" da sequência inteira.

5️⃣ A camada Dense(softmax)

    Usa a saída da LSTM para prever a próxima palavra na sequência.
'''
model_1.fit(dataset, epochs=3)

model_1.save("meu_modelo.keras")
model_1.save_weights('meu_modelo.weights.h5')
################################################


