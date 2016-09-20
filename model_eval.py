'''
Evaluate and validate the final model.
More details in the "Model Evaluation and Validation" section of the report.
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

# Run model evaluation function from the learning_agent
learning_agent.NUM_ENVS = 1
rewards = learning_agent.model_evaluation()

# Print mean and standard deviation of rewards
print('Rewards mean: %f\nRewards std-dev: %f' % (np.mean(rewards), np.std(rewards)))

# Avg reward for final 100 episodes
print('Average reward for final 100 episodes: %f' % np.mean(rewards[-100:]))

# Save the rewards list just in case we need it later
print('Saving rewards to eval_rewards.p')
with open('eval_rewards.p', 'wb') as rewards_out:
	pickle.dump(rewards, rewards_out)

# Plot rewards over episodes for visualization
plt.plot(rewards)
plt.title('Reward over Episodes')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.show()