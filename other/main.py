from __future__ import print_function
import gym
import search_tree as st
import network
import time
import numpy as np

f = open('results.txt','w') 

envs = ['CartPole-v0', 'Pong-ram-v0', 'Roulette-v0']
trainings = [2000, 5000, 10000]
etas = [.1, .2, .4, .8]
search_times = [1, 10, 100, 1000]
reward_scales = [1.0, 1.0, 1.0]
observation_scales = [10, 256, 1]
observation_sizes = [5, 129, 2]
env_index = -1
for temp_env in envs:
  env_index = env_index + 1
  for training in trainings:
    for eta in etas:
      for search_time in search_times:
        env = gym.make(temp_env)
        net = network.Network([observation_sizes[env_index], 200, 200, observation_sizes[env_index]], observation_scales[env_index], reward_scales[env_index], eta)
        tree = st.SearchTree(net, env.action_space.n, search_time)


        #Setting up to learn
        observation = env.reset()
        action = env.action_space.sample()
        total_reward = 0
        total_baseline_reward = 0


        #Training
        for counter in range(training+1000):

          # Construct the x vector
          x_arr = np.append(observation, action)
          i = 0
          x = np.zeros((observation_sizes[env_index],1))
          for item in x_arr:
            x[i][0] = np.array(item)
            i = i + 1

          #Take the selected action
          observation, reward, done, info = env.step(action)

          # Construct the y vector
          y_arr = np.append(observation/net.obs_scale, reward/net.rew_scale)
          i = 0
          y = np.zeros((observation_sizes[env_index],1))
          for item in y_arr:
            y[i][0] = np.array(item)
            i = i + 1


          # Update the network
          tree.update_model([(x,y)])

          if training > counter:
            action = env.action_space.sample()
            total_baseline_reward = total_baseline_reward + reward
          else:
            action = tree.get_action(observation)
            total_reward = total_reward + reward

          #reset the environment if necessary
          if done:
            observation = env.reset()
        
        p_1 = total_reward/1000
        p_2 = total_baseline_reward/training
        f.write("The environment is %s\n" % temp_env)
        f.write("The eta is %f\n" % eta)
        f.write("The time is %f\n" % search_time)
        f.write("The training is %f\n" % training)
        f.write("The total reward is %f\n" % p_1)
        f.write("The total baseline reward is %f\n" % p_2)

f.close()
