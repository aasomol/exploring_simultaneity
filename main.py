"""
	This file is the executable for running PPO. It is based on this medium article: 
	https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gym
import sys
import torch

from ppo import PPO
from structs import FeedForwardNN
from eval_policy import eval_policy

from communication import createTranslator

import argparse


def get_args():
	"""
		Description:
		Parses arguments at command line.

		Parameters:
			None

		Return:
			args - the arguments parsed
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename (not needed for train)
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename (not needed for train)
	parser.add_argument('--num_problem', dest='num_problem', type=int, default=0)      # number of problem for pddlgym
	parser.add_argument('--max_num', dest='max_num', type=int, default=-1)             # The total number of problems that are being analyzed (only for test mode)
	parser.add_argument('--env', dest='pddlgym_env', type=str)						   # The PDDLGym environment name of the domain

	args = parser.parse_args()

	return args

def train( hyperparameters, actor_model, critic_model, num_problem, pddlgym_env):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = PPO(policy_class=FeedForwardNN,num_problem = num_problem, env_id = pddlgym_env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	model.learn(total_timesteps=600_000)

def test(actor_model,num_problem,name_env,max_num):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	translator = createTranslator(name_env,device='cpu',num_problem=num_problem)
	literal_space_size, action_space_size = translator.getSizes()

	obs_dim = literal_space_size + action_space_size
	act_dim = action_space_size

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardNN(act_dim, 1)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))

	# Evaluate our policy 
	eval_policy(policy=policy, env=translator, name=name_env, num=num_problem, max_num=max_num)

def main(args):
	"""
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""

	# Here is where one can set hyperparameters for the PPO algorithm
	hyperparameters = {
				'timesteps_per_batch': 3000,  
				'max_timesteps_per_episode': 500, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 5,
				'lr': 0.02,
				'clip': 0.2
			  }


	# Train or test, depending on the mode specified 
	if args.mode == 'train':
		train(hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model, num_problem=args.num_problem, pddlgym_env=args.pddlgym_env)
	else:
		test(actor_model=args.actor_model, num_problem=args.num_problem, name_env=args.pddlgym_env, max_num=args.max_num)

if __name__ == '__main__':
	# Parse arguments from command line
	args = get_args() 
	main(args)