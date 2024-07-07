import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random

# Path to dataset
dataDir = "C:\\Users\\blacb\\Desktop\\Datasets\\kagglecatsanddogs_5340\\PetImages"

# Categories we will destinguish
categories = ["Dog", "Cat"]

# Train neural network
trainingData = []


for category in categories: #Goes through cat and dog

    path = os.path.join(dataDir, category) # PetImages\Dog , PetImages\Cat

    # 0 for dog, 1 for cat
    classNumber = categories.index(category)

    for img in os.listdir(path):
        
        # Some images are broken, ignore the ones that throw errors
        try:
            # Loads the image and changes it into gray scale
            imageArray = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

            # These show the image
            # plt.imshow(imgArray, cmap = "gray")
            # plt.show()

            # Resize all images to same size, 100x100
            imageSize = 100
            resizedImgArray = cv2.resize(imageArray, (imageSize, imageSize))

            # Adds the image of the animal, along with its corresponding label
            trainingData.append([resizedImgArray, classNumber])

        
        except Exception as e:
            pass
    

# This shuffles the training data up, as otherwise it would just be
# dog dog dog dog dog cat cat cat cat (which would mess things up)
random.shuffle(trainingData)

x_train = [] # The image data of the animal
y_train = [] # The label (what animal it is) 0, 1

for features, label in trainingData: # 2D list so 2 variables needed

    x_train.append(features)
    y_train.append(label)

# (How many features (we don't know so -1), pixel size, pixel size, ray scale (so only one channel))

# We use -1 since we don't know how many images will be passed through (since some will be corrupted)
# If we know that there will be exactly 100 images, then the first value will be 100 (100 different arrays with x pixels and y pixels with a colour depth of 1)
x_train = np.array(x_train).reshape(-1, imageSize, imageSize, 1)

# This saves the training data so that we don't have to rerun this script again and again
np.save("DogCatTrainingData.npy", x_train)
np.save("DogCatLabels.npy", y_train)











        

