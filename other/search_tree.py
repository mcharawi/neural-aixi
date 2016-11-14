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
	

    def update_model(self, x, y):
	""" Update the search tree's model of the env
	with the given  data. 'x' is the concatentation
	of the previous observation, and the action taken
	and 'y' is the concatenation of the new observation
	and the reward."""
	self.model.update([x, y])

    def run_simulation(self, action):
	"""The method that runs a playout simulation for a given
	beginning action."""
	return reward

    def get_action(self):
	"""Get the best possible action given the current
	model of the environment."""
	return action

