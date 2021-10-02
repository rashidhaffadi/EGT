
import torch
import torch.nn as nn
from torch.functional import F
from utils import *

class SequenceModel(nn.Module):
	"""docstring for SequenceModel"""
	def __init__(self, input_size, hidden_size, n_layers, **kwargs):
		super(SequenceModel, self).__init__()
		
		self.rnn = nn.LSTM(input_size, hidden_size, n_layers, **kwargs)
		self.linear = nn.Sequential(linear(hidden_size, hidden_size, True, 0.5), 
									linear(hidden_size, hidden_size, True, 0.5))
		self.fc = nn.Linear(hidden_size, 128)
		
		# load 

	def forward(self, x):
		x, _ = self.rnn(x)
		x = x.mean(1)
		x = self.linear(x)

		return self.fc(x)