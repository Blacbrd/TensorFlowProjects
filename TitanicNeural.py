# Used help from:
# https://www.tensorflow.org/tutorials/load_data/csv

import keras
import numpy as np
import pandas as pd

# Directory for titanic data
dataDirTrain = "C:\\Users\\blacb\\Desktop\\Datasets\\titanic\\train.csv"

# Reads csv file
dataFrameTitanicTrain = pd.read_csv(dataDirTrain)

# Removes survived column and adds it to this variable
labelsCSV = dataFrameTitanicTrain.pop("Survived")

# Numpy arrays to feed into network
trainingFeatures = np.array(dataFrameTitanicTrain)
labels = np.array(labelsCSV)

model = keras.models.Sequential()

model.add(keras.layers.Input(trainingFeatures.shape))

model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))

model.add(keras.layers.Dense(1, activation="sigmoid"))

optimiser = keras.optimizers.Adam(learning_rate=0.001)

# This tells the model to strive for accuracy, therefore to reduce loss
model.compile(optimizer=optimiser, loss=keras.losses.MeanSquaredError(), metrics=["accuracy"])

model.fit(trainingFeatures, labels, epochs=5)







