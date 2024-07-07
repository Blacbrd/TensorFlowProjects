import keras
import numpy as np

# Add in the numpy arrays you created
x_train = np.load("NumpyArrays\\DogCatTrainingData.npy")
y_train = np.load("NumpyArrays\\DogCatLabels.npy")

# Normalise data, between 0 - 1
x_train = keras.utils.normalize(x_train, axis=1)

# Feed forward
model = keras.models.Sequential()

# This is the input layer of the neural network, which takes in the whole image
# Input shape is the width x height x colour depth, 100 x 100 x 1
model.add(keras.layers.Input(shape=x_train.shape[1:]))

# 64 is the amount of filters/kernels we use, each finding different types of patterns
# (3, 3) is the size of the kernel matrix. Can also just use 3
# Strides is how much we want to slide over the image each time, 1,1 just means over by one pixel
# Padding is the values that the kernel will take when it overshoots the image (goes into the void where there is no image)
# Non linearity introduced through relu
model.add(keras.layers.Conv2D(64, (3, 3), strides=(1,1), padding="valid", activation="relu"))

# Pooling reduces the size/resolution of our image while still maintaining important information
# (2,2) essentially divides both of our dimensions by 2, 50 x 50
model.add(keras.layers.MaxPool2D((2,2)))

# Now we add 2 more of the same layers, but remove all the default arguments
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.MaxPool2D((2,2)))

# Make the image into a 1D array, allows us to use pixels as neurone input layer
model.add(keras.layers.Flatten())

# Fully connected layer, similar to regular NN
model.add(keras.layers.Dense(64, activation="relu"))

# Only 1 output needed, high for dog, low for cat or vise versa
# SINCE WE ARE ONLY DEALING WITH 2 CLASSES, WE CAN USE SIGMOID
model.add(keras.layers.Dense(1, activation="sigmoid"))



# Learning rate is how fast the gradient can change to find the local minimum/maximum at a time
optimiser = keras.optimizers.Adam(learning_rate=0.001)

# This tells the model to strive for accuracy, therefore to reduce loss
model.compile(optimizer=optimiser, loss="binary_crossentropy", metrics=["accuracy"])

# Training:
# Batch size is how many images you pass through the network at a time
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
