# DQN for OpenAI Gym environments
#
# additions:
# - vectorized batch updates
# - sliding window experience replay
# - prioritized experience replay proportional to error
# - Double Q network for more stable learning

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.optimizers import rmsprop

import gym
from gym import wrappers

from models import simple, conv

############################################
### Reinforcement Learning Configuration ###
############################################

class Agent():
	def __init__(self, game_string, model_fn, done_reward=0, max_steps_per_episode=100000, monitor=False):
		self.env = gym.make(game_string)
		if monitor:
			self.env = wrappers.Monitor(self.env, '/tmp/experiment', force=True)
		self.num_actions = self.env.action_space.n
		self.state_shape = self.env.reset().shape
		self.max_steps_per_episode = max_steps_per_episode
		self.done_reward = done_reward

		self.model = model_fn(self.num_actions, self.state_shape)
		self.model2 = model_fn(self.num_actions, self.state_shape) # double q network
		self.batch_size = 64 # replays used per learning cycle

		self.epsilon = 1 # chance to take a random action
		self.min_epsilon = .01
		self.epsilon_decay = lambda x: x*.995 # decays through this fn each learning cycle
		self.gamma = .99	# long-term vs short-term reward

		# experience replay
		self.replay_capacity = 100000

		replay_shape = np.array([np.zeros(self.state_shape), 0, 0, np.zeros(self.state_shape), False, 0.0]).shape
		replay_shape = list(replay_shape)
		replay_shape.insert(0, self.replay_capacity)
		replays_shape = tuple(replay_shape)

		self.replays = np.zeros(replays_shape, dtype=object)
		self.replay_n = 0
		self.replays_full = False

		# prioritized proportional experience replay
		self.default_error = 1.0
		self.total_replay_error = 0.0

	#################
	# Model methods #
	#################

	def predictOne(self, state, dqn_n=1):
		if dqn_n == 1:
			return self.model.predict(np.array([state]))[0]
		else:
			return self.model2.predict(np.array([state]))[0]

	#############################
	# Experience replay methods #
	#############################

	def add_replay(self, state, action, reward, state_next, done):
		self.total_replay_error -= self.replays[self.replay_n % self.replay_capacity][5] # remove error from replaced
		self.total_replay_error += self.default_error
		replay = np.array([state, action, reward, state_next, done, self.default_error])
		self.replays[self.replay_n % self.replay_capacity] = replay
		self.replay_n += 1


	def select_replay_indices(self):
		batch_size = min(self.batch_size, self.replay_n)
		n = min(self.replay_n, self.replay_capacity)
		if self.total_replay_error != 0.0:
			replay_errors = self.replays[:n][:, 5]
			prioritization = np.multiply(replay_errors, 1/self.total_replay_error).astype(float)
			return np.random.choice(n, size=batch_size, replace=False, p=prioritization)
		else:
			return np.random.choice(n, size=batch_size, replace=False)

	#######################
	# Acting and Learning #
	#######################

	def get_action(self, state):
		if np.random.rand() < self.epsilon:
			action = self.env.action_space.sample()
		else:
			action = np.argmax(self.predictOne(state))
		return action


	def learn(self):
		# randomly either learn on the first or second model
		if np.random.rand() < .5:
			m1 = self.model
			m2 = self.model2
		else:
			m1 = self.model2
			m2 = self.model

		replay_batch_indices = self.select_replay_indices()

		no_state = np.zeros(self.state_shape)
 
		s = np.array([ self.replays[i][0] for i in replay_batch_indices ])
		s_ = np.array([ (no_state if self.replays[i][3] is None else self.replays[i][3]) for i in replay_batch_indices ])
		
		p = m1.predict(s)
		p_ = m1.predict(s_)
		p2_ = m2.predict(s_)

		i = 0
		for replay_index in replay_batch_indices:
			_, action, reward, _, done, _ = self.replays[replay_index]

			target = reward
			if not done:
				max_value_action = np.argmax(p_[i])
				target += self.gamma*p2_[i][max_value_action]

			# replace error value in replay with updated amount
			new_error = abs(p[i][action] - target)
			self.total_replay_error -= self.replays[replay_index][5]
			self.total_replay_error += new_error
			self.replays[replay_index][5] = new_error

			p[i][action] = target
			i += 1

		m1.fit(s, p, epochs=1, verbose=0)


	def one_episode(self, episode_number, total_episodes):
		state = self.env.reset()

		for t in range(self.max_steps_per_episode):
			self.env.render()

			action = self.get_action(state)
			state_next, reward, done, info = self.env.step(action)

			if done: reward += self.done_reward

			self.add_replay(state, action, reward, state_next, done)
			self.learn()

			state = state_next
			if done:
				break

		print("episode: {}/{}, score: {}, epsilon: {}".format(episode_number, total_episodes, t, self.epsilon))


	def play(self, episodes=4000):
		for episode in range(episodes):
			self.one_episode(episode_number=episode, total_episodes=episodes)
			if self.epsilon > self.min_epsilon:
				self.epsilon = self.epsilon_decay(self.epsilon)

