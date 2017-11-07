from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

one = np.array([1, 0, 0])
two = np.array([0, 1, 0])
three = np.array([0, 0, 1])


seq = np.array([one, two, two, three, three, three])
next_in_seq = np.array([two, two, three, three, three, one])
seq_length = np.size(seq, 0)


model = Sequential()
model.add(LSTM(32, stateful=True, batch_size=1, input_shape=(1, 3)))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer)


for i in range(1000):
	x = np.array([seq[i % seq_length]])
	y = np.array([next_in_seq[i % seq_length]])

	model.fit(
		x=np.array([x]),
		y=np.array(y),
		batch_size=1,
		epochs=1)

model.reset_states()

for i in range(100):
	x = np.array([seq[i % seq_length]])

	# Check if the received value is what you predicted
	if i > 0: # if it's not the first time
		mistake = ''
		if not np.array_equal(x[0], prediction):
			mistake = 'mistake!'
		print('prediction:', prediction, 'actual:', x[0], mistake)

	probabilities = model.predict(np.array([x]))
	prediction = np.array([0, 0, 0])
	prediction[np.argmax(probabilities[0])] = 1