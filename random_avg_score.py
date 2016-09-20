'''
Find the average score over 100 episodes, by taking random actions at each timestep
'''
import gym
import numpy as np

env = gym.make('Acrobot-v1')
observation = env.reset()

# List of rewards after each episode
rewards = []

# Run 100 episodes
for i in range(100):
	total_reward = 0

	while True:
		# Take a random action in the environment
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)

		total_reward += reward

		if done:
			print('Episode %s finished with total reward of %s' % (i+1, total_reward))
			break

	rewards.append(total_reward)
	observation = env.reset()

# Calculate the average reward over 100 episodes
avg_reward = np.mean(rewards)
print('Average reward over 100 episodes: %s' % avg_reward)