# Importing the Keras libraries and packages
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 128
# Step 1 - Building the CNN

# Initializing the CNN
model = Sequential()

# First convolution layer and pooling
model.add(Convolution2D(64, (3, 3), input_shape=(sz, sz, 1), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.1))
# Second convolution layer and pooling
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
# third convolution layer and pooling
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
# If needed.....
# fourth convolution layer and pooling
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
# Flattening the layers
model.add(Flatten())

# Adding a fully connected layer

#model.add(Dense(units=128, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.4))

model.add(Dense(units=27, activation="relu"))
model.add(Dropout(0.5))

# softmax for more than 2 outputs neuron
model.add(Dense(units=27, activation="softmax"))

# Compiling the CNN
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)  # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model
model.summary()

plot_model(model,show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    "myProcessData/train",
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

early_stop = EarlyStopping(monitor="val_loss", 
						mode="min",
						patience=3)

checkpointer = ModelCheckpoint(filepath='img_model.best.hdf5',
				verbose=1,
				save_best_only=True)

					
r=model.fit(
    training_set,
    steps_per_epoch=840,  # No of images in training set
    epochs=5,
    validation_data=test_set,
    validation_steps=360,
	callbacks=[early_stop, checkpointer]
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
# Saving the model
model_json = model.to_json()
with open("model_az.json", "w") as json_file:
    json_file.write(model_json)
print("Model Saved")
model.save_weights("model_az.h5")
print("Model saved")
