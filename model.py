
import torch
import torch.nn as nn
from torch.functional import F

def load_model(output_size, num_of_ngrams=200000,pretrained=False, path="/", name="checkpoint1.state_dict"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceModel().to(device).eval()
    
    if pretrained:
        state_dict = torch.load(path + name, map_location=device)
        model.load_state_dict(state_dict)
        
        model.chars_embedding = nn.Embedding(num_embeddings=num_of_ngrams, padding_idx=0, embedding_dim=embedding_dim)
    return model

class GeneralReLU(nn.Module):
 		"""docstring for GeneralReLU"""
 		def __init__(self, leak=None, sub=None, maxv=None):
 			super(GeneralReLU, self).__init__()
 			self.leak, self.sub, self.maxv = leak, sub, maxv
 		
 		def forward(self, x):
 			x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
 			if self.sub is not None: x.sub_(self.sub)
 			if self.maxv is not None: x.clamp_max_(self.maxv)
 			return x

 class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=1):
        "Output will be 2*sz"
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def conv_layer(ni, nf, ks, stride=2, padding=0, leak=None, sub=None, maxv=None, bn=False, dp=None):
	conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding)
	layers = [conv]
	if bn is not None: layers.append(nn.BatchNorm2d(nf))
	if dp is not None: layers.append(nn.Dropout(dp))
	relu = GeneralReLU(leak, sub, maxv)
	layers.append(relu)
	return nn.Sequential(*layers)

def linear_layer(ni, no, leak=None, sub=None, maxv=None):
	return nn.Sequential(nn.Linear(ni, no), 
						 GeneralReLU(leak, sub, maxv))

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
	def __init__(self, eye_params=[[]], face_params=[[]], face_grid_params=[[]], pretrained=False):
		super(StaticModel, self).__init__()
		self.eye_params = eye_params
		self.face_params = face_params
		self.face_grid_params = face_grid_params

		self.eye_model = EyeModel(self.eye_params)
		self.face_model = FaceModel(self.face_params)
		self.face_grid_model = FaceGridModel(self.face_grid_params)

		self.fc1 = linear_layer(, 128)
		self.fc2 = linear_layer(, 128)
		self.out = nn.Linear(128*2, 2)

	def forward(self, xf, xl, xr, xg)
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


def freeze(m):
	pass

class SequenceModel(nn?Module):
	"""docstring for SequenceModel"""
	def __init__(self, arg):
		super(SequenceModel, self).__init__()
		self.encoder = StaticModel(pretrained=True)
		freeze(self.encoder)

		# load 

	def forward(self, x):
		features = self.encoder(x)
		
		