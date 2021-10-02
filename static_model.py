
import torch
import torch.nn as nn
from torch.functional import F
from utils import *


# params = [[3, 96, 11, 4], ...]

class CNNModel(nn.Module):
    """docstring for EyeMosel"""
    def __init__(self, params=[], **kwargs):
        super(CNNModel, self).__init__()
        self.params = params
        self.layers = [conv_layer(*param, **kwargs) for param in  self.params]
        self.features = nn.Sequential(*self.layers)
        self.pool = AdaptiveConcatPool2d(1)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
#         x = x.view(x.size(0), -1)
        return x
		
# eye_params=[]
class EyeModel(nn.Module):
	"""docstring for EyeModel"""
	def __init__(self, params=[[]]):
		super(EyeModel, self).__init__()
		self.params = params
		self.cnn = CNNModel(self.params[:-1])
		self.fc = linear_layer(*self.params[-1])

	def forward(self, xl, xr):
		xl = self.cnn(xl)
		xr = self.cnn(xr)
		x = torch.cat((xl, xr), 1)
		x = self.fc(x)
		return x
		
class FaceModel(nn.Module):
	"""docstring for FaceModel"""
	def __init__(self, params=[[]]):
		super(FaceModel, self).__init__()
		self.params = params
		self.cnn = CNNModel(self.params[:-1])
		self.fc = linear_layer(*params[-1])
	def forward(self, x):
		x = self.cnn(x)
		x = self.fc(x)
		return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, params=[[]]):
        super(FaceGridModel, self).__init__()
        self.params = params
        self.cnn = CNNModel(self.params[:-1])
        self.fc = linear_layer(*params[-1])

    def forward(self, x):

        x = self.cnn(x)
        x = self.fc(x)
        return x

					
class StaticModel(nn.Module):
	"""docstring for EncoderCNN"""
	def __init__(self, eye_params=[[]], face_params=[[]], face_grid_params=[[]]):
		super(StaticModel, self).__init__()
		self.eye_params = eye_params
		self.face_params = face_params
		self.face_grid_params = face_grid_params

		self.eye_model = EyeModel(self.eye_params)
		self.face_model = FaceModel(self.face_params)
		self.face_grid_model = FaceGridModel(self.face_grid_params)

		self.fc1 = linear_layer(256+64, 128)
		self.fc2 = linear_layer(256+128, 128)
		self.out = nn.Linear(128, 2)

	def freeze(self):
		self.requires_grad_(False)
	def unfreeze(self):
		self.requires_grad_(True)


	def forward(self, xf, xl, xr, xg):
		# eyes
		xe = self.eye_model(xl, xr)#out: 1024
		#face and  grid
		xf = self.face_model(xf)#out: 512
		xg = self.face_grid_model(xg)#out: 64
		xf = torch.cat((xf, xg), 1)
		xf = self.fc1(xf)

		x = torch.cat((xe, xf), 1)
		x = self.fc2(x)
		x = self.out(x)

		return x
