import torch 
import os 
from torchvision import models
from torch.nn import init
import numpy as np 
from torch import nn 
def weight_initialsation(m):
    classname=m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.normal_(m.weight.data,0.0,0.02)
    elif classname.find("Linear")!=-1:
        init.normal_(m.weight.data,0.0,0.2)
    elif classname.find("BatchNorm2d")!=-1:
        init.normal_(m.weight.data,0.1,0.2)
        init.constant(m.bias.data,0.0) 
def xavier_initialisation(m):
    classname=m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.xavier_normal_(m.weight.data,gain=0.02)
    elif classname.find("Linear") !=-1:
        init.xavier_normal_(m.weight.data,gain=0.02)
    elif classname.find("BatchNorm2d")!=-1:
        init.xavier_normal_(m.weight.data,gain=0.02)
        init.constant(m.bias.data,0.0)                     