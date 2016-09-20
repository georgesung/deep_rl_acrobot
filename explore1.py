'''
Exploration:
Run one episode of acrobot by uniformly sampling random actions.
Prints the observation, action, and reward at each timestep.
'''
import gym

env = gym.make('Acrobot-v1')

observation = env.reset()

t = 0
while True:
	old_obs = observation
	action = env.action_space.sample()
	observation, reward, done, info = env.step(action)

	print('obs: %s, action: %s, reward: %s' % (old_obs, action, reward))

	t += 1
	if done:
		print('Episode finished after %d timesteps' % t)
		break
