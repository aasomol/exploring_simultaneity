"""
	Thanks to Eric Yu for valuable insights about PPO implementation. Part of this code is based on his implementation.
	See: https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
	We assume GPU is available. If not, just change the device 'cuda:0' to 'cpu'.
"""

from communication import createTranslator
from structs import Proportional, FeedForwardNN

import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from torch_geometric.utils import softmax as scattered_softmax
from torch_geometric.utils import scatter 

class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self,
        policy_class,
		num_problem: int = 0,
        env_id : str = "",
        **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)
		self.env_id = env_id
		self.num_problem = num_problem

		# Create a translator, which is the object that mediates with PDDLGym. It is an envelope for a PDDLGym env.
		print(env_id)
		self.translator = createTranslator(env_id,'cuda:0',self.num_problem)
		literal_space_size, action_space_size = self.translator.getSizes()

		self.obs_dim = literal_space_size + action_space_size
		self.act_dim = action_space_size

		 # Initialize actor and critic networks
		self.actor = policy_class(self.act_dim, 1).to(torch.device('cuda:0'))
		self.critic = policy_class(self.obs_dim, 1).to(torch.device('cuda:0'))

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		self.init_time = time.time_ns()
		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

		# Initialize sampler for exploration
		self.sampler = Proportional(entropy_flag=True)


	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far

		early_stopping = last_avg_ep_lens = new_avg_ep_lens = 0

		while t_so_far < total_timesteps:                                                                       

			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_glob_acts,batch_glob_index = self.rollout() 

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			V, _, _ = self.evaluate(batch_obs, batch_acts, batch_glob_acts, batch_glob_index, calculate_logs = False)
			A_k = batch_rtgs - V.detach()                                                                      

			# Advantage normalization
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                 

				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs, entropy_loss = self.evaluate(batch_obs, batch_acts, batch_glob_acts, batch_glob_index)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses and adding entropy loss.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				actor_loss = actor_loss - 0.01 * entropy_loss

				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# For early stopping 
			if i_so_far == 0:
				pass
			else:
				if len(self.logger['batch_lens']) > 0 : 
					new_avg_ep_lens = np.mean(self.logger['batch_lens'])
				else:
					new_avg_ep_lens = 0

				if abs(last_avg_ep_lens - new_avg_ep_lens < 0.05) and new_avg_ep_lens < 100:
					early_stopping += 1
				else:
					early_stopping = 0

			last_avg_ep_lens = new_avg_ep_lens

			# Print a summary of our training so far
			self._log_summary()


			# Save our model if it's time
			if i_so_far % self.save_freq == 0 or early_stopping > 10:
				torch.save(self.actor.state_dict(), f'./trained_models/{self.env_id}/{self.num_problem}_actor.pth')
				torch.save(self.critic.state_dict(), f'./trained_models/{self.env_id}/{self.num_problem}_critic.pth')

			# For statistics
			if early_stopping > 10:
				with open(f'./trained_models/{self.env_id}/timesteps.txt','a') as f:

					t= time.time_ns()
					t = time.time_ns() - self.init_time
					t = t / 1e9

					print(self.logger['t_so_far'], t,file=f)
				return


	def rollout(self):
		"""
			This is where we collect the batch of data from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)

				batch_glob_acts - the actions taken into account in this batch. Shape: (number of timesteps, number of actions, dimension of action)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		batch_glob_acts = []
		batch_glob_index = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode

			# Reset the environment. Note that obs is short for observation. 
			obs = self.translator.reset()
			done = False

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):

				t += 1 # Increment timesteps ran this batch so far


				# Calculate action and make a step in the env. 
				action, log_prob , all_acts = self.get_action(obs)

				if action is not None:
					# Track observations in this batch
					batch_obs.append(obs)

					obs, rew, done, _ = self.translator.step(all_acts[action],verbose=False,ep_t=ep_t)
					# Track recent reward, action, and action log probabilities
					ep_rews.append(rew)
					batch_acts.append(action+len(batch_glob_acts))
					batch_log_probs.append(log_prob)

					batch_glob_acts += all_acts
					batch_glob_index += [t-1 for i in all_acts]

					# If the environment tells us the episode is terminated, break
					if done:
						break
				# If there is no applicable action available, break
				else:
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)


		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float, device='cuda:0')
		batch_acts = torch.tensor(batch_acts, dtype=torch.int, device='cuda:0')
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device='cuda:0')
		batch_rtgs = self.compute_rtgs(batch_rews).to(torch.device('cuda:0'))                                                             


		batch_glob_acts = torch.tensor(batch_glob_acts, dtype=torch.int, device='cuda:0')
		batch_glob_index = torch.tensor(batch_glob_index, dtype=torch.int64, device='cuda:0')


		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_glob_acts, batch_glob_index

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs

	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		actions = self.translator.get_all_current_actions()
		if actions:
			actions_num = [self.translator.get_number_of_action(a) for a in actions]
			actions_repr = torch.cat([self.translator.get_action_repr(a).unsqueeze(dim=0) for a in actions],dim=0)

			# raw values
			pi = self.actor(actions_repr)
			PI = F.softmax(pi,dim=0).view(pi.numel())

			if PI.numel() == 1:
				action, log_prob = 0,torch.log(PI[0].view(1))
			else:
				action, log_prob = self.sampler.sample(pi.detach())
			
			return action, log_prob.detach(), actions_num
		else:
			return None, 0, -1

	def evaluate(self, batch_obs, batch_acts, batch_glob_acts, batch_glob_index,calculate_logs=True):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()

		
		if calculate_logs:

			# Discrete log-probabilities calculation:

			all_batch_vectors = torch.index_select(self.translator.action_space_repr,0,batch_glob_acts)

			pi = self.actor(all_batch_vectors)

			# Softmax using pytorch geometric
			pi = scattered_softmax(pi.squeeze(),batch_glob_index)

			# For the entropy
			logpi= -pi.log()

			mul = torch.mul(pi,logpi)

			H = scatter(mul,batch_glob_index,reduce='sum')


			pi = pi[batch_acts].view(batch_acts.numel())
			
			# Return the value vector V of each observation in the batch
			# and log probabilities log_probs of each action in the batch,
			# along with the entropy
			
			return V, pi.log(), H.mean()
		else:
			return V, None, 0

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 3000                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 500           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.99                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean().cpu() for losses in self.logger['actor_losses']])

		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []

