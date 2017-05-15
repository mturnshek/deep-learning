# Reaches around 88% test accuracy on CIFAR-10
# Originally optimized for CIFAR-3 (12000 training images of types cat, boat, automobile)

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import rmsprop
from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import cifar10

################################
### Load and manipulate data ###
################################

num_classes = 10

(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

#######################
### Configure model ###
#######################

# Uses idea from https://arxiv.org/pdf/1412.6806.pdf
# All convolutional, with stride 2 layers replacing max-pooling

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), strides=2, padding='same', activation='relu'))
model.add(Dropout(0.33))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.50))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.50))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

optimizer = rmsprop(lr=0.0001, decay=1e-6)

model.compile(optimizer=optimizer,
	loss='categorical_crossentropy',
	metrics=['accuracy'])

model.summary() # display a summary of the model

#######################
### Train, evaluate ###
#######################

# fit the model on batches with real-time data augmentation
datagen = ImageDataGenerator(
	rotation_range=12,
	horizontal_flip=True,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.1)
datagen.fit(train_data)

epochs_per_evaluation = 10
evaluations = 300

def train_and_evaluate(i, epochs=epochs_per_evaluation):
	print("\n", i, "iterations (10 epochs/iteration)")
	model.fit_generator(datagen.flow(train_data,
									 train_labels,
									 batch_size=64),
	                    steps_per_epoch=len(train_data) / 64,
	                    epochs=epochs)
	evaluation = model.evaluate(test_data, test_labels, batch_size=64, verbose=1)
	print("\n", model.metrics_names)
	print(evaluation)

for i in range(evaluations):
	train_and_evaluate(i)