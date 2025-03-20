#!/usr/bin/python3
#https://lejdiprifti.com/2023/10/14/creating-a-text-generation-neural-network-with-tensorflow/

################################################################################
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()

tokenizer = tokenizer_from_json(tokenizer_json)
total_words = len(tokenizer.word_index) + 1

################################################################################
input_shape=0
with open('input_shape.json', 'r') as archivo_json:
    datos_cargados = json.load(archivo_json)
    input_shape = datos_cargados["input_shape"]
    
################################################################################
import mymodules.model as mmm
import tensorflow as tf

model = mmm.get_model(total_words, input_shape)
model.load_weights('meu_modelo.weights.h5')  # Carregue os pesos


# model = tf.keras.models.load_model("meu_modelo.keras")
# input_shape = model.input_shape[1]

################################################################################
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

seed_text = "April is"
next_words = 5

for Id in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    token_list = pad_sequences([token_list], maxlen=input_shape, padding='pre')
    predicted  = np.argmax(model.predict(token_list), axis=-1)

    output_word = tokenizer.index_word[predicted[0]]

    seed_text += " " + output_word
    
    print(Id,':',seed_text)
