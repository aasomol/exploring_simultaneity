"""
	This file is used only to evaluate our trained policy/actor against the problem that was trained for
"""
from pddlgym import core
import torch
from structs import Proportional
import numpy as np
import torch.nn.functional as F


def get_action(translator, sampler, policy, obs,env):
	"""
		Queries an action from the actor network, should be called from rollout.

		Parameters:
			obs - the observation at the current timestep

		Return:
			action - the action to take, as a numpy array
			log_prob - the log probability of the selected action in the distribution
	"""
	actions = translator.get_all_current_actions()
	actions_num = [translator.get_number_of_action(a) for a in actions]
	actions_repr = torch.tensor(np.array([translator.get_action_repr(a) for a in actions]), dtype=torch.float)
	
	# raw values
	if actions:
		pi = policy(actions_repr)
		
		PI = F.softmax(pi,dim=0).view(pi.numel())

		argmax = torch.argmax(PI).int()

		if PI.numel() == 1:
			action, log_prob = 0,torch.log(PI[0].view(1))
		else:
			action, log_prob = sampler.sample(pi.detach())


		return action, actions_num
	return None, None


def get_best_path(policy, env):
	"""
		Returns a plan to the problem using a policy.

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
		
		Return:
			best_path - The best plan for the problem following the policy

	"""
	tries = 10
	best_len = float('inf')
	best_path = ""
	for i in range(tries):
		obs = env.reset()
		done = False
		path=""
		
		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		while not done and t < 500:
			t += 1

			# Query deterministic action from policy and run it
			action,actions_num = get_action(env,Proportional(entropy_flag=True),policy,obs,env)
			if action is not None:
				
				obs, rew, done, _ = env.step(actions_num[action],1)
				path += str(env.get_action(actions_num[action])) + '\n'

				# Sum all episodic rewards as we go along
				ep_ret += rew
			else:
				break

		# Track episodic length
		ep_len = t

		if ep_len < best_len and t < 100:
			best_path = path
			best_len = ep_len
			best_path += 'done'

	return best_path

def eval_policy(policy, env, name, num, max_num):
	"""
		Function that tries to produce a plan using the trained policy.

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""

	# Get best plan
	path = get_best_path(policy,env)

	# Save it to file
	files = [str(a) for a in range(max_num+1)]
	files.sort()

	with open('plans/'+name+f'/{files[num]}.pddl','w') as f:
		f.write(path)