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

# Wont be relevant in the data
dataFrameTitanicTrain.pop("PassengerId")
dataFrameTitanicTrain.pop("Cabin")

# Removes survived column and adds it to this variable
labelsCSV = dataFrameTitanicTrain.pop("Survived")

# New training data without survived column
trainingDataCSV = dataFrameTitanicTrain

# Need to fill in null values, otherwise "np.unique()" will not work 
trainingDataCSV["Embarked"] = trainingDataCSV["Embarked"].fillna("N")
# trainingDataCSV["Cabin"] = trainingDataCSV["Cabin"].fillna("NULL")


# Since each column has a different datatype and such, we need to split them up
# Initialise a dictionary which asigns the name of the column to a keras Tensor (layer)
inputs = {}

for name, column in trainingDataCSV.items():

    dataType = column.dtype

    if dataType == object: # If its a string

        dataType = tf.string
    
    else:

        dataType = tf.float32 # Numerical value

    # Shape = (1,) is a scalar, extra comma used to define as a tuple
    # Name is just the name assigned to the layer
    # For example, 'age' : kerasTensor shape(None, 1), dtype=float32 etc.
    inputs[name] = keras.Input(shape=(1,), name=name, dtype=dataType)


# Now, we need to normalise all numerical inputs:

# This is an example of dictionary comprehension in python
# What it does, is it creates a new dictionary with this syntax:
# {key: value for key, value in iterable if condition}
# Essentially, it goes over the keys and values of the other dictionary we made,
# and then makes a new dictionary by adding all keys and values that have a dType of float32
numericInputs = {name : input for name, input in inputs.items() if input.dtype == tf.float32}

# The concatonate layer is used to join multiple layers into one layer
# This is usually done with inputs, allowing to have multiple different input layers passed in as one
# We create an array of the values (all tensors) in the numericInput dictionary
x = keras.layers.Concatenate()(list(numericInputs.values()))

# Normalises all the values in the columns with the specified names in the pandas table
# Adapt works out the mean and varience, which is required for normalisation
# Normalisation plots all numbers on a normal curve with mean 0 and variance of 1
normalise = keras.layers.Normalization()
normalise.adapt(np.array(dataFrameTitanicTrain[numericInputs.keys()]))

allNumericInputs = normalise(x)

preprocessedInputs = [allNumericInputs]

# Now we have to encode all of our string inputs, as neural networks can only deal with numerical values
for name, input in inputs.items(): 

    # Skip over numeric inputs
    if not input.dtype == tf.string:

        continue

    # This creates numeric representation of all string values, for example, male -> 3 and female -> 5, arbitrary
    # trainingDataCSV[name] returns the column for the name, if name == "Age", then it returns the values of the age column and their index
    # We use unique, as lets say name == "Sex", then the only two values we have are male and female, and so vocabulary would be ["male", "female"]

    stringLookup = keras.layers.StringLookup(vocabulary=np.unique(trainingDataCSV[name]))
    oneHot = keras.layers.CategoryEncoding(num_tokens=stringLookup.vocabulary_size())

    # String lookup gives us the numeric data
    # One hot gives us the one shot array of the data, eg [0,1] for male, [1,0] for female
    x = stringLookup(input)
    x = oneHot(x)

    # Appends the now numeric string data to the previous numeric data 
    preprocessedInputs.append(x)

# Concatonates/joins all the layers together into one big input layer
preprocessedInputsConcatonated = keras.layers.Concatenate()(preprocessedInputs)


# Essentially, everything above was to set up this model.
# This model takes the data, and preprosses it through a network and layers
# It takes the raw inputs from the pandas dataframe, and then outputs the nicely processed inputs
# These inputs are in one layer, normalised, and in numerical form
preprocessingModel = keras.Model(inputs, preprocessedInputsConcatonated)


# Now we need to create a dictionary of tensors, this is to allow for flexible representation
# of complex relationships, especially since we're dealing with multiple different types of data
# We need to define this since keras doesn't know how to represent your pandas dataframe
titanicDataDict = {name : np.array(value)
                   for name, value in trainingDataCSV.items()}



def titanicModel(preprocessingHead, inputs):

    body = keras.Sequential([
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    preprocessedInputs = preprocessingHead(inputs)
    result = body(preprocessedInputs)
    model = keras.Model(inputs, result)

    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

    return model

titanicModel = titanicModel(preprocessingModel, inputs)

titanicModel.fit(x=titanicDataDict, y=labelsCSV, epochs=20)

titanicModel.save("titanicSurvival.keras")


# SO APPARENTLY, NNs ARE NOT GOOD FOR SMALL DATASETS
# NEED TO USE A DIFFERENT MODEL!!!

