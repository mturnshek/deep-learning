import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.optimizers import rmsprop

import numpy as np

import gym

###########################
### Model Configuration ### 
###########################

def simpleNN(num_actions, state_shape):
	model = Sequential()

	model.add(Dense(32, input_shape=state_shape, activation='relu'))
	model.add(Dense(32, activation='relu'))

	# Max of this will be the action we take
	model.add(Dense(num_actions, activation='linear'))

	optimizer = rmsprop(lr=0.0002, decay=1e-6)
	model.compile(optimizer=optimizer, loss='mse')

	model.summary()

	return model

############################################
### Reinforcement Learning Configuration ###
############################################

class Agent():
	def __init__(self, game_string, model_fn, done_reward=0, max_steps_per_episode=2000):
		self.env = gym.make(game_string)
		self.num_actions = self.env.action_space.n
		self.state_shape = self.env.reset().shape
		self.max_steps_per_episode = max_steps_per_episode
		self.done_reward = done_reward

		self.model = model_fn(self.num_actions, self.state_shape)
		self.replays = []
		self.replay_decay_size = 2048
		self.batch_size = 32

		# q-learning parameters
		self.epsilon = 1
		self.min_epsilon = .01
		self.epsilon_decay = lambda x: x - .001
		self.gamma = .99	# long-term vs short-term reward

	def add_replay(self, state, action, reward, state_next, done):
		self.replays.append((state, action, reward, state_next, done))

	# halves replays at random ... meant to weight more recent memories more highly
	def replay_decay(self):
		kept_indices = np.random.choice(len(self.replays), len(self.replays)//2, replace=False)
		kept_replays = []
		for replay_index in kept_indices:
			kept_replays.append(self.replays[replay_index])
		self.replays = kept_replays

	def get_action(self, state):
		if np.random.rand() < self.epsilon:
			action = self.env.action_space.sample()
		else:
			action = np.argmax(self.model.predict(state)[0])
		return action

	def learn(self):
		batch_size = min(self.batch_size, len(self.replays))
		replay_indices = np.random.choice(len(self.replays), batch_size)

		states = []
		target_fs = []

		for replay_index in replay_indices:
			state, action, reward, state_next, done = self.replays[replay_index]
			if done:
				target = reward
			else:
				next_prediction = np.amax(self.model.predict(state_next)[0])
				target = reward + self.gamma*(next_prediction)

			target_f = self.model.predict(state)
			target_f[0][action] = target

			states.append(state[0])
			target_fs.append(target_f[0])

		self.model.fit(np.array(states), np.array(target_fs), epochs=1, verbose=0)

	def one_episode(self, episode_number, total_episodes):
		state = self.env.reset()
		state = np.array([state])

		for t in range(self.max_steps_per_episode):
			self.env.render()

			action = self.get_action(state)
			state_next, reward, done, info = self.env.step(action)
			state_next = np.array([state_next])

			if done:
				reward += self.done_reward
			
			self.add_replay(state, action, reward, state_next, done)

			if len(self.replays) >= self.replay_decay_size:
				self.replay_decay()

			self.learn()

			state = state_next
			if done:
				break

		print("episode: {}/{}, score: {}, epsilon: {}".format(episode_number, total_episodes, t, self.epsilon))

	def play(self, episodes=1000):
		for episode in range(episodes):
			self.one_episode(episode_number=episode, total_episodes=episodes)
			if self.epsilon > self.min_epsilon:
				self.epsilon = self.epsilon_decay(self.epsilon)

##############################
### Create and run learner ###
##############################

agent = Agent('CartPole-v0', simpleNN, done_reward=-100)

agent.play()