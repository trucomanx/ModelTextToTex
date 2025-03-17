#https://lejdiprifti.com/2023/10/14/creating-a-text-generation-neural-network-with-tensorflow/

import tensorflow as tf

import mymodules.dataset as mmds

# List os strings
training_data = mmds.load_wiki_dataset(max_len=50000)

print("len(training_data)",len(training_data))
print(training_data[1])


################################################################################

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(training_data)

total_words = len(tokenizer.word_index) + 1

#print(tokenizer.word_index) # dictionary key=word value=index
print(total_words)


input_sequences = []
for single_line in training_data:
  # transform each sentence into a sequence of integers
  token_list = tokenizer.texts_to_sequences([single_line])[0]
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)

print("len(input_sequences):",len(input_sequences))
print("input_sequences[0]:",input_sequences[0])


import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 75% das sequências terão tamanho menor ou igual a esse valor
max_sequence_len = np.int32(np.percentile([len(x) for x in input_sequences], 75))

# padd the input_sequences until the max_sequence_len
# padding='pre': O preenchimento acontece antes da sequência, ou seja, adiciona zeros no início
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))


# xs = input_sequences[:,:-1] - recebe todas as linhas (:) e todas as colunas exceto a última (:-1) da matriz input_sequences. 
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

print("padding sequences shape:",input_sequences.shape) # (791292, 18)
print("               xs shape:",xs.shape) # (791292, 17)
print("           labels shape:",labels.shape) # (791292, )

# build the dataset with batches of 512 and autotuned prefetch
dataset = tf.data.Dataset.from_tensor_slices((xs, labels)).batch(512).prefetch(tf.data.AUTOTUNE)

print(dataset)

import mymodules.model as mmm

model_1 = mmm.get_model(total_words)


model_1.compile(loss=tf.losses.SparseCategoricalCrossentropy(), 
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                metrics=['accuracy'])

model_1.summary()

################################################
'''
Como o modelo "entende" que são sequências?

1️⃣ Formato dos dados de entrada

    Seu xs tem dimensão (N, 17), ou seja, cada amostra tem 17 números (índices de palavras).
    Como passamos todas as palavras de uma vez para o modelo, ele vê cada amostra como um grupo de palavras relacionadas.

2️⃣ A camada Embedding

    Converte cada número (índice de palavra) em um vetor denso de tamanho 128.
    Resultado: A entrada, que antes era (17,), vira um tensor (17, 128).
    Agora, temos uma sequência de vetores e não só números.

3️⃣ A camada LSTM

    O LSTM lê os vetores da sequência um por um, na ordem, e processa a relação entre eles.
    Como a camada mantém um estado interno (memória) ao longo do tempo, ela entende a sequência como algo conectado, e não apenas como números soltos.

4️⃣ A saída da segunda LSTM (32 neurônios)

    Retorna um único vetor de tamanho 32, que contém a "memória" da sequência inteira.

5️⃣ A camada Dense(softmax)

    Usa a saída da LSTM para prever a próxima palavra na sequência.
'''
model_1.fit(dataset, epochs=5)


################################################

seed_text = "April is"
next_words = 10

for _ in range(next_words):
 token_list = tokenizer.texts_to_sequences([seed_text])[0]
 token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
 predicted = np.argmax(model_1.predict(token_list), axis=-1)
 output_word = ""
 for word, index in tokenizer.word_index.items():
  if index == predicted:
   output_word = word
   break
 seed_text += " " + output_word
print(seed_text)
