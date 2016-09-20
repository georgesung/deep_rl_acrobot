'''
Searching for the optimal parameter combinations.
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

import learning_agent

# Configure the parameter search
NUM_ITERS = 20  # how many iterations of random parameter search
NUM_REPEAT = 3  # how many times to repeat the learning process for a given parameter set
REPORT_FILE = 'search_params.csv'  # save results to this file

# Lists of hyper-parameter values on which to perform random search (w/o replacement)
ACTOR_LR = [0.05, 0.01, 0.005, 0.001, 0.0005]  # 0.005 -> 0.0005
CRITIC_LR_SCALE = [1.25, 1.0, 0.75, 0.5, 0.25]
REWARD_DISCOUNT = [0.99, 0.97, 0.95, 0.93]
A_REG_SCALE = [0.005, 0.0005, 0.00005]
C_REG_SCALE = [0.005, 0.0005, 0.00005]

# Helper functions
def choose_params(results):
	'''
	Choose random parameter combination w/o replacement
	results is the dict that maps (param1, param2, ...) --> score
	Returns tuple of chosen parameters
	'''
	# Keep trying random parameter combinations until we get a unique combination
	while True:
		actor_lr = np.random.choice(ACTOR_LR)
		critic_lr_scale = np.random.choice(CRITIC_LR_SCALE)
		reward_discount = np.random.choice(REWARD_DISCOUNT)
		a_reg_scale = np.random.choice(A_REG_SCALE)
		c_reg_scale = np.random.choice(C_REG_SCALE)

		params = (actor_lr, critic_lr_scale, reward_discount, a_reg_scale, c_reg_scale)

		if params not in results:
			break

	return params

def set_params(params):
	'''
	Sets the parameters specified in params tuple to the learning_agent
	'''
	learning_agent.ACTOR_LR = params[0]
	learning_agent.CRITIC_LR_SCALE = params[1]
	learning_agent.REWARD_DISCOUNT = params[2]
	learning_agent.A_REG_SCALE = params[3]
	learning_agent.C_REG_SCALE = params[4]


##########################################
# Main script to perform parameter search
##########################################

# Dictionary to store results
# (param1, param2, ...) --> score
results = {}

# Write csv header of report file
report = open(REPORT_FILE, 'w')
report.write('ACTOR_LR,CRITIC_LR_SCALE,REWARD_DISCOUNT,A_REG_SCALE,C_REG_SCALE,score\n')
report.close()

for _ in range(NUM_ITERS):
	params = choose_params(results)  # choose random parameter combination w/o replacement
	set_params(params)  # set the parameters

	for _ in range(NUM_REPEAT):
		# Run RL algorithm until it does not return an error
		while True:
			avg_rewards, score = learning_agent.run_rl()
			if avg_rewards is not None:
				break

		# Calculate the score, i.e. average reward of final 100 episodes
		score = np.mean(avg_rewards[-100:])

		# Save the max score for this parameter combination
		if params in results:
			if score > results[params]:
				results[params] = score
		else:
			results[params] = score

	# Append results to report file
	report = open(REPORT_FILE, 'a')
	report.write('%f,%f,%f,%f,%f,%f\n' % (*params, score))
	report.close()

