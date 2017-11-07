from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding
from keras.layers import GRU
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys


def one_hot_encode(indices_array, n_classes):
	one_hot_targets = np.eye(n_classes)[indices_array]
	return one_hot_targets


def one_hot_encode_single(index, n_classes):
	one_hot_vector = np.zeros(n_classes)
	one_hot_vector[index] = 1
	return one_hot_vector


### Data preprocessing
path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

text_y = list(text[1:])
text_x = list(text[:-1])

text_x = np.array(text_x)
text_y = np.array(text_y)

text_indices_x = np.zeros(text_x.shape, dtype='uint32')
text_indices_y = np.zeros(text_y.shape, dtype='uint32')

for i in range(0, text_indices_x.size):
	text_indices_x[i] = char_indices[text_x[i]]
	text_indices_y[i] = char_indices[text_y[i]]

x = one_hot_encode(text_indices_x, len(chars))
y = one_hot_encode(text_indices_y, len(chars))

x = x.reshape(x.shape[0], 1, x.shape[1])
y = y.reshape(y.shape[0], 1, y.shape[1])

### Build model
training_batch_size = 30
training_model = Sequential()
training_model.add(GRU(128, batch_input_shape=(training_batch_size, x.shape[1], x.shape[2]), return_sequences=True, stateful=True))
training_model.add(Activation('relu'))
training_model.add(GRU(128, return_sequences=True, stateful=True))
training_model.add(Activation('relu'))
training_model.add(Dense(len(chars)))
training_model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
training_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


### Batch train training_model
epochs = 10
for i in range(epochs):
	training_model.fit(x=x, y=y, batch_size=training_batch_size, epochs=1, shuffle=False)
	training_model.reset_states()


### Predict single characters at a time with a stateful model
prediction_batch_size = 1
prediction_model = Sequential()
prediction_model.add(GRU(128, batch_input_shape=(prediction_batch_size, x.shape[1], x.shape[2]), return_sequences=True, stateful=True))
prediction_model.add(Activation('relu'))
prediction_model.add(GRU(128, return_sequences=True, stateful=True))
prediction_model.add(Activation('relu'))
prediction_model.add(Dense(len(chars)))
prediction_model.add(Activation('softmax'))
prediction_model.set_weights(training_model.get_weights())

while True:
	user_input = input('? > ')
	if len(user_input) == 1:
		# retrieve index from char
		x_index = char_indices[user_input]
		# encode index
		encoded = one_hot_encode_single(x_index, len(chars))
		# reshape encoding so it can go in the net
		x = encoded.reshape(1, 1, encoded.shape[0])
		# get the prediction
		y = prediction_model.predict(x)
		# get the new char's index from the prediction
		y_index = np.argmax(y.reshape(x.size))
		# get the new char from the index
		y_char = indices_char[y_index]
		print(y_char)