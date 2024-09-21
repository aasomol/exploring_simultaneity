from typing import Type

import numpy as np
import torch as th
import torch.nn as nn

import torch.nn.functional as F

class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = th.tensor(obs, dtype=th.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output

class Proportional:
    def __init__(self, entropy_flag=False):
        self.entropy_flag = entropy_flag
        
    def sample(self,pi):
        pi = F.softmax(pi.squeeze(),dim=0)
        dist = th.distributions.categorical.Categorical(pi.squeeze())
        action = dist.sample()
        return action.item(), dist.log_prob(action)