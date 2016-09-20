'''
Given the optimal parameter combination, run 1500 episodes of training from scratch,
followed by 8500 episodes of training with the learning rate reduced by a factor of 10

More details in the "Refinement" section of the report.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt
import pickle

import learning_agent

# Set parameters in learning_agent
learning_agent.ACTOR_LR = 0.05
learning_agent.CRITIC_LR_SCALE = 0.5
learning_agent.REWARD_DISCOUNT = 0.97
learning_agent.A_REG_SCALE = 0.00005
learning_agent.C_REG_SCALE = 0.0005

# Enable saving the model to disk
learning_agent.SAVE_MODEL = True

###################################################################
# Phase 1: Run training over 1500 episodes, save the trained model
###################################################################

# Configure learning_agent to run 1500 training episodes, from scratch
learning_agent.NUM_EPISODES = 1500
learning_agent.RESUME = False

# Run RL algorithm until it does not return an error
while True:
	avg_rewards1, score1 = learning_agent.run_rl()
	if avg_rewards1 is not None:
		break

print('Phase 1 complete, score: %f' % score1)

###################################################################
# Phase 2: Load model from Phase 1, reduce the learning rate by
# a factor of 10, run another 8500 training episodes
###################################################################

# Configure learning_agent appropriately
learning_agent.NUM_EPISODES = 8500
learning_agent.RESUME = True
learning_agent.ACTOR_LR /= 10

# Run RL algorithm until it does not return an error
while True:
	avg_rewards2, score2 = learning_agent.run_rl()
	if avg_rewards2 is not None:
		break

print('Phase 2 complete, final score: %f' % score2)
print('Final model saved at %s' % learning_agent.MODEL_LOC)

avg_rewards = np.concatenate((avg_rewards1, avg_rewards2))

# Save the avg_rewards list just in case we need it later
# Maybe our plot was unclear, and we need to re-plot w/ same data
print('Saving avg_rewards to avg_rewards.p')
with open('avg_rewards.p', 'wb') as avg_rewards_out:
	pickle.dump(avg_rewards, avg_rewards_out)

print('Plotting avg rewards over episodes')
plt.plot(avg_rewards)
plt.title('Average Reward over Episodes')
plt.ylabel('Average Reward')
plt.xlabel('Episode')
plt.show()
