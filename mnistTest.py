import tensorflow as tf

mnist = tf.keras.datasets.mnist # 28x28 hand written digits

# Unpack dataset into training and testing data

# x is the actual image, y is the label assigned to the image
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalises data to make it between 0 and 1, easier to compute
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Feed forward network, output of prev layer used as input for next
model = tf.keras.models.Sequential()

# Input layer, flattens the images to be a single line rather than a grid
model.add(tf.keras.layers.Flatten())

# Hidden layers, dense means fully connected (neuron count, activation func)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Output layer, makes outputs between 0, 1
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# This says what sort of back propegation to add
model.compile(optimizer= "adam", 
              loss="sparse_categorical_crossentropy",
              metrics = ["accuracy"])

model.fit(x_train, y_train, epochs = 4)

# Tells us how close our model accuracy is to the training data value
value_loss, value_accuracy = model.evaluate(x_test, y_test)
print(value_loss, value_accuracy)