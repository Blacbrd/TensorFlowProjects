# Used help from:
# https://www.tensorflow.org/tutorials/load_data/csv

import keras
import tensorflow as tf
import numpy as np
import pandas as pd

# Directory for titanic data
dataDirTrain = "C:\\Users\\blacb\\Desktop\\Datasets\\titanic\\train.csv"

# Reads csv file
dataFrameTitanicTrain = pd.read_csv(dataDirTrain)

# Removes survived column and adds it to this variable
labelsCSV = dataFrameTitanicTrain.pop("Survived")

# New training data without survived column
trainingDataCSV = dataFrameTitanicTrain

# Since each column has a different datatype and such, we need to split them up
# Initialise a dictionary which asigns the name of the column to a keras Tensor (layer)
inputsToNN = {}

for name, column in trainingDataCSV.items():

    dataType = column.dtype

    if dataType == object: # If its a string

        dataType = tf.string
    
    else:

        dataType = tf.float32 # Numerical value

    # Shape = (1,) is a scalar, extra comma used to define as a tuple
    # Name is just the name assigned to the layer
    # For example, 'age' : kerasTensor shape(None, 1), dtype=float32 etc.
    inputsToNN[name] = keras.Input(shape=(1,), name=name, dtype=dataType)

# Now, we need to normalise all numerical inputs:

# This is an example of dictionary comprehension in python
# What it does, is it creates a new dictionary with this syntax:
# {key: value for key, value in iterable if condition}
# Essentially, it goes over the keys and values of the other dictionary we made,
# and then makes a new dictionary by adding all keys and values that have a dType of float32
numericInputs = {name : input for name, input in inputsToNN.items()
                 if input}

# The concatonate layer is used to join multiple layers into one layer
# This is usually done with inputs, allowing to have multiple different input layers passed in as one
# We create an array of the values (all tensors) in the numericInput dictionary
x = keras.layers.Concatenate()(list(numericInputs.values()))

# Normalises all the values in the columns with the specified names in the pandas table
# Adapt works out the mean and varience, which is required for normalisation
normalise = keras.layers.Normalization()
normalise.adapt(np.array(dataFrameTitanicTrain[numericInputs.keys()]))

allNumericInputs = normalise(x)



