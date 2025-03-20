#!/usr/bin/python3
#https://lejdiprifti.com/2023/10/14/creating-a-text-generation-neural-network-with-tensorflow/

import mymodules.dataset as mmds


# List os strings
training_data = mmds.load_wiki_dataset(max_len=50000)

Ltotal = len(training_data)


print("\ntraining_data[0:5]:")
[print(l+1,'of',Ltotal,':',text) for l,text in enumerate(training_data[0:5])]
print("")

################################################################################


import tensorflow as tf
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(training_data)

total_words = len(tokenizer.word_index) + 1
#print(tokenizer.word_index) # dictionary key=word value=index
print("total_words:",total_words)
print("")


import json

tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(json.loads(tokenizer_json), f, indent=4, ensure_ascii=False)
    

################################################################################

input_sequences = []
for single_line in training_data:
  # transform each sentence into a sequence of integers
  token_list = tokenizer.texts_to_sequences([single_line])[0]
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)

Lseq = len(input_sequences)

print("\ninput_sequences[0:5]:")
[print(l+1,'of',Lseq,':',text) for l,text in enumerate(input_sequences[0:5])]
print("")



import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 75% das sequências terão tamanho menor ou igual a esse valor
Lengths = [len(x) for x in input_sequences]
print("seq len min :", min(Lengths))
print("seq len mean:", np.mean(Lengths))
print("seq len std :", np.std(Lengths))
print("seq len max :", max(Lengths))


import matplotlib.pyplot as plt


plt.hist(Lengths, bins=40, edgecolor='black')
plt.xlabel('Valores')
plt.ylabel('Frequência')
plt.title('Histograma')
plt.show()

max_sequence_len = np.int32(np.percentile(Lengths, 95))
print("max_sequence_len:", max_sequence_len)
print("")

# padd the input_sequences until the max_sequence_len
# padding='pre': O preenchimento acontece antes da sequência, ou seja, adiciona zeros no início
input_pad_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

print("\ninput_pad_sequences[0:5]:")
[print(l+1,'of',Lseq,':',text) for l,text in enumerate(input_pad_sequences[0:5])]
print("")


# xs = input_pad_sequences[:,:-1] - recebe todas as linhas (:) e todas as colunas exceto a última (:-1) da matriz input_pad_sequences. 
xs, labels = input_pad_sequences[:,:-1],input_pad_sequences[:,-1]

print("padding sequences shape:",input_pad_sequences.shape) # (791292, 18)
print("               xs shape:",xs.shape) # (791292, 107)
print("           labels shape:",labels.shape) # (791292, )
print("")

# build the dataset with batches of 512 and autotuned prefetch
dataset = tf.data.Dataset.from_tensor_slices((xs, labels)).batch(512).prefetch(tf.data.AUTOTUNE)

# Caminho para salvar
path = "meu_dataset"

# Salvando com o método atualizado
dataset.save(path)
