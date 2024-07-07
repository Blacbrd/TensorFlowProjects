import tensorflow as tf
import numpy as np

# Add in the numpy arrays you created
x_train = np.load("NumpyArrays\\DogCatTrainingData.npy")
y_train = np.load("NumpyArrays\\DogCatLabels.npy")

# Normalise data, between 0 - 1
x_train = tf.keras.utils.normalize(x_train, axis=1)

# Feed forward
model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Conv2D(64), (3, 3), input_shape = x_train.shape[1:])


