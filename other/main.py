import gym
import search_tree as st
import network
import time
import numpy as np

env = gym.make('SpaceInvaders-ram-v0')
network = network.Network([129, 200, 200, 200, 129], 256.0, 100.0, 0.1)
st = st.SearchTree(network, env.action_space.n, 1)


#Setting up to learn
observation = env.reset()
action = env.action_space.sample()
total_reward = 0
total_baseline_reward = 0
training = 5000

#Training
for counter in range(10000):

  # Construct the x vector
  x_arr = np.append(observation, action)
  i = 0
  x = np.zeros((129,1))
  for item in x_arr:
    x[i][0] = np.array(item)
    i = i + 1

  #Take the selected action
  observation, reward, done, info = env.step(action)

  # Construct the y vector
  y_arr = np.append(observation/network.obs_scale, reward/network.rew_scale)
  i = 0
  y = np.zeros((129,1))
  for item in y_arr:
    y[i][0] = np.array(item)
    i = i + 1


  # Update the network
  st.update_model([(x,y)])

  if training < counter:
    action = env.action_space.sample()
    total_baseline_reward = total_baseline_reward + reward
  else:
    action = st.get_action(observation)
    total_reward = total_reward + reward

  #reset the environment if necessary
  if done:
    observation = env.reset()


print("The total reward is %d" % total_reward)
print("The total baseline reward is %d" % total_baseline_reward)
