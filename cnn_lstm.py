
import torch
import torch.nn as nn
from torch.functional import F
from utils import *
from static_model import *
from temporal_model import *
from global_variables import *



# self.head = nn.Sequential(nn.Linear(//??, 128), 
# 								  nn.Linear(128, 2))

def load_encoder(path='./models/', name="checkpoint1.state_dict", **kwargs):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StaticModel(eye_params, face_params, face_grid_params).to(device).eval()
    state_dict = torch.load(path + name, map_location=device)
    model.load_state_dict(state_dict)

    return nn.Sequential(*list(model.children())[:-1])

class CNNtoRNN(nn.Module):
	"""docstring for CNNtoRNN"""
	def __init__(self, ins, hs, nl, ):
		super(CNNtoRNN, self).__init__()
		# load encoder
		# train sequence model
		# concatenate + fc
		# out
		self.encoder = load_encoder(**kwargs)
		self.encoder.freeze()
		self.decoder = SequenceModel(**kwargs)

		self.fc = linear(128*2, 128, True, 0.5)
		self.out = nn.Linear(128, 2)

	def forward(self, xf, xl, xr, xg):
		xe = self.encoder(xf, xl, xr, xg)
		xd = self.decoder(xe)
		x = torch.cat((xd, xe), 1)

		x = self.fc(x)
		return self.out(x)