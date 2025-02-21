import numpy as np
import tensorflow as tf
from tensorflow import keras # need to open vs code from conda environment
from tensorflow.keras import layers
import io

# Setting up models with layers and training choices

def train_one_var_model(data, labels):
    normalizer = layers.Normalization(input_shape=[1,], axis=None)
    normalizer.adapt(data)
    
    model = tf.keras.Sequential([
         normalizer,
         layers.Dense(units=1)
     ])

    model.compile(
         optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
         loss='mean_absolute_error')

    history = model.fit(
         data,
         labels,
         epochs=100,
         verbose=0,
         validation_split = 0.2)
    
    return model

def train_multi_var_model(data, labels):
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(data)
    
    model = tf.keras.Sequential([
         normalizer,
         layers.Dense(units=1)
     ])

    model.compile(
         optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
         loss='mean_absolute_error')

    history = model.fit(
         data,
         labels,
         epochs=100,
         verbose=0,
         validation_split = 0.2)
    
    return model

def train_dnn_model(data, labels):
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(data)
    
    model = tf.keras.Sequential([
         normalizer,
         layers.Dense(64, activation='relu'),
         layers.Dense(64, activation='relu'),
         layers.Dense(1)
     ])

    model.compile(
         optimizer=tf.keras.optimizers.Adam(0.001),
         loss='mean_absolute_error')

    history = model.fit(
         data,
         labels,
         epochs=100,
         verbose=0,
         validation_split = 0.2)
    
    return model

# Functions working on the models

def get_model_summary(model, include_weights):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    if include_weights == True:
         for layer in model.layers:
               summary_string += "Layer: " + layer.name + "\n"
               summary_string += "   weights: " + str(layer.get_weights()[0]) + "\n"
               summary_string += "   biases: " + str(layer.get_weights()[1]) + "\n"
    return summary_string

def infer(model, data):
   return model.predict(data)

def get_weights(model, layer):
   return np.array(model.get_layer(layer).get_weights()[0])

def get_bias(model, layer):
   return np.array(model.get_layer(layer).get_weights()[1])

