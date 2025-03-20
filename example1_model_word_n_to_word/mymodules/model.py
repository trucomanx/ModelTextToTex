import tensorflow as tf


def get_model(total_words):
    inputs = tf.keras.layers.Input(shape=(1,))

    x = tf.keras.layers.Embedding(total_words, 128, mask_zero=False)(inputs)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x =  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)

    outputs = tf.keras.layers.Dense(total_words, activation="softmax", name="output_layer")(x)

    model_1 = tf.keras.models.Model(inputs, outputs)


    
    return model_1
