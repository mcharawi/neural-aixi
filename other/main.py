import gym
import network
import time
import numpy as np

env = gym.make('AirRaid-ram-v0')
network = network.Network([129, 200, 200, 200, 129], 256.0, 100.0)
observation = env.reset()
env.render()
for counter in range(10000):
	
	# Select a random action
	action = env.action_space.sample()
	
	# Construct the x vector
	x_arr = np.append(observation, action)
	i = 0
	x = np.zeros((129,1))
	for item in x_arr:
		x[i][0]= np.array(item)
		i= i + 1
	
	#Take the random action
        observation, reward, done, info = env.step(action)
	
	# COnstruct the y vector
	y_arr = np.append(observation/network.obs_scale, reward/network.rew_scale)
	i = 0	
	y = np.zeros((129,1))
	for item in y_arr:
		y[i][0] = np.array(item)
		i = i + 1
	
	# Update the network
	network.update_mini_batch([(x,y)], 0.1)
	
	#reset the environment if necessary
        if done:
	    observation = env.reset()
