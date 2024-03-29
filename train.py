# Importing the Keras libraries and packages
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import os
import matplotlib.pyplot as plt
sz = 128
# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(64, (3, 3), input_shape=(sz, sz, 1), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Convolution2D(64, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# third convolution layer and pooling
classifier.add(Convolution2D(64, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())
=
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dropout(0.5))

classifier.add(Dense(units=27, activation="relu"))
classifier.add(Dropout(0.5))

# softmax for more than 2 outputs neuron
classifier.add(Dense(units=27, activation="softmax"))

# Compiling the CNN
classifier.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)  # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model
classifier.summary()

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    "myProcessdata/train",
    target_size=(sz, sz),
    batch_size=5,
    color_mode="grayscale",
    class_mode="categorical",
)

test_set = test_datagen.flow_from_directory(
    "myProcessData/test",
    target_size=(sz, sz),
    batch_size=5,
    color_mode="grayscale",
    class_mode="categorical",
)
r=classifier.fit_generator(
    training_set,
    steps_per_epoch=1000,  # No of images in training set
    epochs=15,
    validation_data=test_set,
    validation_steps=405,
)  # No of images in test set

#plotting loss
plt.title('Cross Entropy Loss')
plt.plot(r.history['loss'], color='blue', label='train')
plt.plot(r.history['val_loss'], color='orange', label='test')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
#plotting accuracy
plt.title('Accuracy')
plt.plot(r.history['accuracy'], color='blue', label='train')
plt.plot(r.history['val_accuracy'], color='orange', label='test')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
#1

# Saving the model
model_json = classifier.to_json()
with open("model_az.json", "w") as json_file:
    json_file.write(model_json)
print("Model Saved")
classifier.save_weights("model_az.h5")
print("Model saved")
