'''
Implementing policy gradient actor/critic reinforcement learning method ("Deep Deterministic Policy Gradients"),
with parallel environments ("Asynchronous Methods for Deep Reinforcement Learning").

See "Implementation" section of the report for more details.
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

# Environment parameters
ENV = 'Acrobot-v1'
OBS_WIDTH = 6
NUM_ACTIONS = 3

# Save/restore previously trained models
RESUME = False  # resume from previously trained model?
SAVE_MODEL = False  # save final trained model?
MODEL_LOC = 'models/model.ckpt'  # final model save location
SAVE_THRESHOLD = -200  # must achieve score above this threshold to save model

# Save results here to upload to OpenAI Gym leaderboard
# Only applicable in model evaluation phase
RECORD_LOC = 'openai_data'

# Overall parameters
NUM_EPISODES = 1000
MAX_ITER = 3000  # max number of timesteps to run per episode
NUM_ENVS = 5  # number of environments to run in parallel
# NOTE: Not using replay buffer, so set below two parameters to value 1
EPISODES_PER_UPDATE = 1  # i.e. how many episodes per replay buffer
DS_FACTOR = 1  # replay buffer downsample factor (num_samples = buffer_size // DS_FACTOR)

# Model hyper-parameters
ACTOR_LR = 0.005  # actor network learning rate
CRITIC_LR_SCALE = 0.5  # scaling factor of critic network learning rate, relative to actor
CRITIC_LR = ACTOR_LR * CRITIC_LR_SCALE  # do not tune this parameter, tune the above
REWARD_DISCOUNT = 0.97
A_REG_SCALE = 0.0005  # actor network regularization strength
C_REG_SCALE = 0.0005  # critic network regularization strength

########################################
# Helper functions
########################################
def discount_rewards(r):
	'''
	Take 1D float array of rewards and compute discounted reward
	Slightly modified from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
	'''
	discounted_r = np.zeros_like(r)
	running_add = 0.
	for t in reversed(range(len(r))):
		running_add = running_add * REWARD_DISCOUNT + r[t]
		discounted_r[t] = running_add
	return discounted_r

def sample_action(probs):
	'''
	Sample action (0/1/2/etc.) from probability distribution probs
	'''
	num_actions = len(probs)
	threshold = random.uniform(0,1)
	cumulative_prob = 0.
	for action in range(num_actions):
		cumulative_prob += probs[action]
		if cumulative_prob > threshold:
			return action
	return num_actions - 1  # might need this for strange corner case?

########################################
# Actor and Critic networks
########################################
def actor_network():
	'''
	Actor network, including policy gradient equation and optimizer
	'''
	with tf.variable_scope('policy'):
		# Inputs
		state = tf.placeholder('float', [None, OBS_WIDTH])  # batch_size x obs_width
		actions = tf.placeholder('float', [None, NUM_ACTIONS])  # batch_size x num_actions
		advantages = tf.placeholder('float', [None, 1])  # batch_size x 1

		# 3-layer fully-connected neural network
		mlp_out = slim.stack(state, slim.fully_connected, [6, NUM_ACTIONS], weights_regularizer=slim.l2_regularizer(scale=A_REG_SCALE))

		# Network output
		probabilities = tf.nn.softmax(mlp_out)

		good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions), reduction_indices=[1])
		eligibility = tf.log(good_probabilities) * advantages

		# Loss & optimizer
		data_loss = -tf.reduce_sum(eligibility)
		reg_losses = slim.losses.get_regularization_losses(scope='policy')
		reg_loss = tf.reduce_sum(reg_losses)
		total_loss = data_loss + reg_loss

		optimizer = tf.train.AdamOptimizer(ACTOR_LR).minimize(total_loss)

		return probabilities, state, actions, advantages, optimizer

def critic_network():
	'''
	Critic network, including loss and optimizer
	'''
	with tf.variable_scope('value'):
		# Inputs
		state = tf.placeholder('float', [None, OBS_WIDTH])  # batch_size x obs_width
		newvals = tf.placeholder('float', [None, 1])  # batch_size x 1

		# 4-layer fully-connected neural network
		calculated = slim.stack(state, slim.fully_connected, [6, 6, 1], weights_regularizer=slim.l2_regularizer(scale=C_REG_SCALE))

		# Error value
		diffs = calculated - newvals

		# Loss & optimizer
		data_loss = tf.nn.l2_loss(diffs)
		reg_losses = slim.losses.get_regularization_losses(scope='value')
		reg_loss = tf.reduce_sum(reg_losses)
		total_loss = data_loss + reg_loss

		optimizer = tf.train.AdamOptimizer(CRITIC_LR).minimize(total_loss)

		return calculated, state, newvals, optimizer, total_loss

########################################
# Training and inference processes
########################################
def train_networks(replay_buffer, actor, critic, sess):
	'''
	Run training on a random subset of experiences in replay buffer
	Arguments:
		replay_buffer: 2D array-like of the form
			[(states, actions, advantages, update_vals)_0, (states, actions, advantages, update_vals)_1, ...]
	'''
	actor_calculated, actor_state, actor_actions, actor_advantages, actor_optimizer = actor
	critic_calculated, critic_state, critic_newvals, critic_optimizer, critic_loss = critic

	# Down-sample the replay buffer
	training_batch_size = len(replay_buffer) // DS_FACTOR
	training_batch = np.array(replay_buffer)[np.random.choice(len(replay_buffer), training_batch_size, False)]

	# "Un-zip" training_batch
	states, actions, advantages, update_vals = list(zip(*training_batch))

	print('Average advantage: %s' % np.mean(advantages))

	# Train critic network (i.e. value network)
	update_vals_vector = np.expand_dims(update_vals, axis=1)
	sess.run(critic_optimizer, feed_dict={critic_state: states, critic_newvals: update_vals_vector})

	# Train actor network (i.e. policy network)
	advantages_vector = np.expand_dims(advantages, axis=1)
	sess.run(actor_optimizer, feed_dict={actor_state: states, actor_advantages: advantages_vector, actor_actions: actions})
# END train_networks

def run_episode(envs, actor, critic, sess):
	'''
	Run a single episode
	'''
	# Actor and critic networks
	actor_calculated, actor_state, actor_actions, actor_advantages, actor_optimizer = actor
	critic_calculated, critic_state, critic_newvals, critic_optimizer, critic_loss = critic

	# Reset env
	observation = [env.reset() for env in envs]

	# Total undiscounted reward for each env
	totalreward = [0 for _ in range(NUM_ENVS)]

	# States, actions, rewards across all timesteps in episode, across all envs
	states = []
	actions = []
	rewards = []

	# Keep track of which envs are done
	done_mask = [False for _ in range(NUM_ENVS)]

	# Interact with the environment
	for _ in range(MAX_ITER):
		# Actor network calculates policy
		probs = sess.run(actor_calculated, feed_dict={actor_state: observation})

		# Sample action from stochastic policy
		action = [sample_action(prob) for prob in probs]

		# Record state and action if applicable. Record None object if particular env is already done.
		states.append([observation[i] if not done_mask[i] else None for i in range(NUM_ENVS)])

		action_onehot = [np.zeros(NUM_ACTIONS) for _ in range(NUM_ENVS)]
		for i in range(NUM_ENVS):
			action_onehot[i][action[i]] = 1
		actions.append([action_onehot[i] if not done_mask[i] else None for i in range(NUM_ENVS)])

		# Reset envs that are already done
		for i in range(NUM_ENVS):
			if done_mask[i]:
				envs[i].reset()

		# Take action in each environment, and store the feedback
		# If env is already done, we will ignore it's result later
		observation, reward, done, info = list(zip(*[envs[i].step(action[i]) for i in range(NUM_ENVS)]))

		# Record the reward, but record None if env is already done
		rewards.append([reward[i] if not done_mask[i] else None for i in range(NUM_ENVS)])

		# Check which env(s) are done in this iteration
		for i in range(NUM_ENVS):
			if done[i]:
				done_mask[i] = True

		# If all envs are done, break
		if all(done_mask):
			break

	# Convert states, actions, and rewards tensor w/ shape num_iters x NUM_ENVS into NUM_ENVS x num_iters (i.e. matrix transpose)
	states_per_env = list(zip(*states))
	actions_per_env = list(zip(*actions))
	rewards_per_env = list(zip(*rewards))

	# For all envs, for all applicable timesteps, do necessary calculations to add this experience
	experiences = []
	for env_idx in range(NUM_ENVS):
		# Some envs finished earlier than others, so remove the None objects in lists
		filtered_states =  [s for s in states_per_env[env_idx] if s is not None]
		filtered_actions = [a for a in actions_per_env[env_idx] if a is not None]
		filtered_rewards = [r for r in rewards_per_env[env_idx] if r is not None]

		# Compute discounted rewards for this env
		disc_rewards = discount_rewards(filtered_rewards)

		# Critic network computes the estimated value of the state/observation
		baseline = sess.run(critic_calculated,feed_dict={critic_state: filtered_states}).ravel()

		# Advantage: How much better is the observed discounted reward vs. baseline computed by critic
		advantages = disc_rewards - baseline

		# Record this experience
		experiences += zip(filtered_states, filtered_actions, advantages, disc_rewards)

		# For book-keeping, record total undiscounted reward for this env
		totalreward[env_idx] = sum(filtered_rewards)

	return totalreward, experiences
# END run_episode

########################################
# Top-level RL algorithm
########################################
def run_rl():
	'''
	Run the reinforcement learning process
	'''
	# Manage the TensorFlow context, i.e. the default graph and session
	with tf.Graph().as_default(), tf.Session() as sess:
		# Create multiple parallel environments, per "Asynchronous Methods" paper
		envs = [gym.make(ENV) for _ in range(NUM_ENVS)]

		# "Instantiate" actor/critic networks
		actor = actor_network()
		critic = critic_network()

		# TF saver to save/restore model
		saver = tf.train.Saver()

		# Initialize or restore model
		if RESUME:
			print('Restoring model from %s' % MODEL_LOC)
			saver.restore(sess, MODEL_LOC)
		else:
			sess.run(tf.initialize_all_variables())

		# Variables for book-keeping and replay buffer
		total_reward = 0.
		avg_rewards = []
		max_avg_reward = None
		replay_buffer = []

		# Run the reinforcement learning algorithm
		for episode_count in range(NUM_EPISODES):
			reward, experiences = run_episode(envs, actor, critic, sess)

			# If more than 1/2 of the envs exceeded MAX_ITER timesteps,
			# generally this means the learning process becomes unstable, since we have incomplete experiences
			# Return None to indicate such an error
			exceed_count = 0
			for r in reward:
				if r <= -MAX_ITER:
					exceed_count += 1
			if exceed_count > NUM_ENVS//2:
				print('ERROR: More than 1/2 of envs exceeded %s timesteps, aborting this run' % MAX_ITER)
				return None, None

			# If no such error as above, proceed
			total_reward += np.mean(reward)
			replay_buffer += experiences
			print('Episode %s, reward (max_iter=%s): %s' % (episode_count, MAX_ITER, reward))

			if (episode_count+1) % EPISODES_PER_UPDATE == 0:
				avg_reward = total_reward / EPISODES_PER_UPDATE
				avg_rewards.append(avg_reward)
				print('Average reward for past %s episodes: %s' % (EPISODES_PER_UPDATE, avg_reward))

				# If the average reward is good enough, then save the model
				if max_avg_reward is None:
					max_avg_reward = avg_reward

				if avg_reward > max_avg_reward:
					max_avg_reward = avg_reward
					print('New max average reward')

					if avg_reward > SAVE_THRESHOLD and SAVE_MODEL:
						save_path = saver.save(sess, MODEL_LOC)
						print('Model saved in file: %s' % save_path)

				print('Running training after episode %s...' % str(episode_count+1))
				train_networks(replay_buffer, actor, critic, sess)
				print('Training complete')

				total_reward = 0.
				replay_buffer = []

		# Calculate final score
		# Run 100 episodes, no training steps, and only look at a single envs[0]
		print('Calculating final score (avg reward over 100 episodes)')
		score = 0.0
		for episode_count in range(100):
			reward, experiences = run_episode(envs, actor, critic, sess)
			score += reward[0]
		score /= 100
		print('Final score: %f' % score)

		# Save final model
		if SAVE_MODEL:
			save_path = saver.save(sess, MODEL_LOC)
			print('Final model saved in file: %s' % save_path)

		# Return list of average total rewards
		return avg_rewards, score

########################################
# Model evaluation
# Only run this after model is trained
########################################
def model_evaluation():
	'''
	Run 1000 consecutive episodes using a pre-trained model
	Also runs the OpenAI Gym environment monitor, so upload results
	The OpenAI Gym results are stored in RECORD_LOC

	Returns a list of total rewards over 1000 episodes
	'''

	# Rewards over episodes
	rewards = []

	with tf.Graph().as_default(), tf.Session() as sess:
		# Similar code as run_rl(), except we only have 1 env, and perform no training
		envs = [gym.make(ENV)]  # create a list of only 1 env

		actor = actor_network()
		critic = critic_network()

		print('Restoring model from %s' % MODEL_LOC)
		saver = tf.train.Saver()
		saver.restore(sess, MODEL_LOC)

		print('Running 1000 episodes. Recording experiment data at %s' % RECORD_LOC)

		envs[0].monitor.start(RECORD_LOC, force=True)  # start OpenAI Gym environment monitor

		for episode_count in range(1000):
			reward, experiences = run_episode(envs, actor, critic, sess)
			rewards.append(reward[0])

		envs[0].monitor.close()  # close OpenAI Gym environment monitor

	return rewards

########################################
# If executing this python file stand-alone
########################################
if __name__ == '__main__':
	# Run RL algorithm until it does not return an error
	while True:
		avg_rewards, score = run_rl()

		if avg_rewards is not None:
			break

	# Plot average rewards over each batch
	print('Plotting avg rewards over episodes')
	plt.plot(avg_rewards)
	plt.ylabel('Average Reward')
	plt.xlabel('Episode')
	plt.show()
