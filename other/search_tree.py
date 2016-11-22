"""
search_tree.py
~~~~~~~~

A class to implement a monte carlo search tree with random
simulations of a given length of time for each action
space. The model of the environment used is a 
feedforward neural network that takes in a descritipn 
of the environment and an action and returns an expected 
observation as well as the expected reward. This network is 
specified in network.py.
"""


import random
import numpy as np
import time

current_milli_time = lambda: int(round(time.time() * 1000))

class SearchTree(object):

    def __init__(self, model, action_space_size, time):
	""" The 'model' is the model of the environment that
	the search tree uses to run simulations and choose
	the best actions. The 'action space size' is the
	number of actions that are possible in a 
	given environment. The 'time' is the number of
	milliseconds to run the simulation for."""
	self.model = model
	self.simulation_time = time
	self.action_space_size = action_space_size
	

    def update_model(self, x):
	""" Update the search tree's model of the env
	with the given  data. 'x' is the concatentation
	of the previous observation, and the action taken
	and 'y' is the concatenation of the new observation
	and the reward."""
	self.model.update(x)

    def run_simulation(self, action, observation):
	"""The method that runs a playout simulation for a given
	beginning action."""
	input_array = np.append(observation, action)
	i = 0
	x = np.zeros((129,1))
	for item in input_array:
		x[i][0] = np.array(item)
		i = i + 1

	out = self.model.feedforward(input_array)
	reward = out[-1][0]
	start_time = current_milli_time()
	while current_milli_time() - start_time < self.simulation_time:
		out = out * 255
		out[-1][0] = random.randint(0, self.action_space_size-1)
		out = self.model.feedforward(out)
		reward = reward + 100*out[-1][0]
	return reward

    def get_action(self, observation):
	"""Get the best possible action given the current
	model of the environment."""
	rewards = []
	for action in range(self.action_space_size):
		rewards.append(self.run_simulation(action, observation))
	return np.argmax(rewards)



