'''
Exploration:
Run 10 episodes of acrobot by uniformly sampling random actions.
Save all observations in memory, and plot a histogram for all 6 dimensions of the observation.
'''
import gym
import numpy as np
import matplotlib.pyplot as plt

# Helper function to plot numpy histogram
def plot_hist(hist_bins, ylabel, filename):
	hist, bins = hist_bins
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2

	plt.bar(center, hist, align='center', width=width)
	plt.ylabel(ylabel)
	plt.xlabel('Value')
	#plt.show()
	plt.savefig(filename, bbox_inches='tight')
	plt.clf()

env = gym.make('Acrobot-v1')
observation = env.reset()

# List of all observations across 10 episodes
observations = []

# Run 10 episodes
for i in range(10):
	t = 0
	while True:
		observations.append(observation)

		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)

		t += 1
		if done:
			env.reset()
			print('Episode finished after %d timesteps' % t)
			break

# 'observations' is an Nx6 matrix, where N is the total number of observations
# To create the histograms, take a transpose of observations so we get a 6xN matrix instead
observations_t = np.transpose(observations)

# For each dimension of the observation, create a histogram
# np.histogram returns (hist, bins)
hist_bins = [np.histogram(dim, bins=20) for dim in observations_t]

# Plot a histogram for each dimension of the observations
dim_num = 1
for hb in hist_bins:
	plot_hist(hb, 'Dimension %d' % dim_num, 'dim%d.png' % dim_num)
	dim_num += 1
