import tensorflow as tf


def get_model(total_words, input_size, output_size):
    inputs = tf.keras.layers.Input(shape=(input_size,))

    # Camada de embedding
    x = tf.keras.layers.Embedding(total_words, 64, mask_zero=False)(inputs)

    # Camada LSTM bidirecional
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    
    # Outra camada LSTM bidirecional, ajustando o número de unidades
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16 * output_size))(x)

    # Camada densa
    x = tf.keras.layers.Dense(16 * output_size, activation="tanh")(x)
        
    # Camada densa
    x = tf.keras.layers.Dense(output_size * total_words, activation="softmax")(x)

    # Reshape para o formato correto
    outputs = tf.keras.layers.Reshape((output_size, total_words))(x)

    model_1 = tf.keras.models.Model(inputs, outputs)
    
    return model_1

'''

def get_model(total_words,input_size,output_size):
    inputs = tf.keras.layers.Input(shape=(input_size,))

    x = tf.keras.layers.Embedding(total_words, 8, mask_zero=False)(inputs)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(x)
    x =  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16*output_size), name="BI_LSTM_16")(x)
    
    x = tf.keras.layers.Reshape((output_size,-1))(x)  

    outputs = tf.keras.layers.Dense(total_words, activation="softmax", name="output_layer")(x)

    model_1 = tf.keras.models.Model(inputs, outputs)


    
    return model_1
'''

