# Using Deep Reinforcement Learning to Solve Acrobot
This project uses policy gradients with actor/critic networks (see "Deep Deterministic Policy Gradients", https://arxiv.org/abs/1509.02971) and parallel environments (see "Asynchronous Methods for Deep Reinforcement Learning", https://arxiv.org/abs/1602.01783) to solve OpenAI Gym's Acrobot-v1 environment. As of September 20, 2016, the final learned model placed 3rd on the OpenAI Gym Acrobot-v1 leaderboard, with a score of -80.69 ± 1.06: https://gym.openai.com/envs/Acrobot-v1

This project is my capstone project for Udacity's Machine Learning Engineer Nanodegree. For the full capstone project report, please see 'Report.pdf'.

## Dependencies
The following depenencies are required:

* Python 2.7/3.5+
* NumPy
* Matplotlib
* OpenAI Gym
* TensorFlow 0.10.0

## How to run
To run the learning agent with pre-set parameter values, run 'python learning_agent.py'. The main reinformcent learning code is located in this file.

To run the parameter search, run 'python search_params.py'. In this file, you can modify the parameter values over which to search.

Once you know your optimal parameters, enter them in 'full_training.py', and run 'python full_training.py'. This will perform the full training process on the model.

To validate your model (make sure results are consistent), run 'python model_eval.py'.

# Detailed report
A full detailed report can be found at 'Report.pdf'
