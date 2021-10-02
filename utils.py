
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
    def __init__(self, leak=None, sub=None, maxv=None, **kwargs):
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
    layers = [nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding)]
    if bn is not None: layers.append(nn.BatchNorm2d(nf))
    if dp is not None: layers.append(nn.Dropout(dp))
    relu = GeneralReLU(leak, sub, maxv)
    layers.append(relu)
    return nn.Sequential(*layers)

def linear_layer(ni, no, leak=None, sub=None, maxv=None):
    return nn.Sequential(nn.Linear(ni, no), 
                         GeneralReLU(leak, sub, maxv))

def linear(ni, no, bn=False, dp=None, **kwargs):
    layers = [nn.Linear(ni, no)]
    if bn: layers.append(nn.BatchNorm1d(no))
    if dp is not None: layers.append(nn.Dropout(dp))
    layers += [GeneralReLU(**kwargs)]
    return nn.Sequential(*layers)