import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.optimizers import rmsprop

############################
### Model Configurations ### 
############################

def simple(num_actions, state_shape):
	model = Sequential()

	model.add(Dense(64, activation='relu', input_shape=state_shape))

	# Max of this will be the action we take
	model.add(Dense(num_actions, activation='linear'))

	optimizer = rmsprop(lr=0.00025, decay=1e-6)
	model.compile(optimizer=optimizer, loss='mse')

	model.summary()

	return model

def conv(num_actions, state_shape):
	model = Sequential()

	model.add(Conv2D(16, (8, 8), strides=4, padding='same', activation='relu', input_shape=state_shape))
	model.add(Conv2D(32, (4, 4), strides=2, padding='same', activation='relu'))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())

	model.add(Dense(num_actions, activation='linear'))

	optimizer = rmsprop(lr=0.0001, decay=1e-6)
	model.compile(optimizer=optimizer, loss='mse')

	model.summary()

	return model