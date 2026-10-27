from mpmath import ker
from torch.nn import ReLU
import torch 
import os 
from torchvision import models
from torch.nn import init
import numpy as np 
from torch import nn 
def weight_init_normal(m):
    classname=m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.normal_(m.weight.data,0.0,0.02)
    elif classname.find("Linear")!=-1:
        init.normal_(m.weight.data,0.0,0.2)
    elif classname.find("BatchNorm2d")!=-1:
        init.normal_(m.weight.data,0.1,0.2)
        init.constant(m.bias.data,0.0) 
def weight_init_xavier(m):
    classname=m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.xavier_normal_(m.weight.data,gain=0.02)
    elif classname.find("Linear") !=-1:
        init.xavier_normal_(m.weight.data,gain=0.02)
    elif classname.find("BatchNorm2d")!=-1:
        init.xavier_normal_(m.weight.data,gain=0.02)
        init.constant(m.bias.data,0.0)      
def weight_init_kaiming(m):
    clasname=m.__class__.__name__
    if clasname.find("Conv")!=-1:
        init.kaiming_normal_(m.weight.data,a=0,mode="fan_in")
    elif clasname.find("Linear"):
        init.kaiming_normal_(m.weight.data,a=0,mode='fan_in')
    elif clasname.find('BatchNorm2d')!=-1:
        init.kaiming_normal_(m.weight.data,1.0,0.02)
        init.constant(m.bias.data,0.0)
def init_weight(net,init_type='normal'):
    print("Initailsation Methos [%s]:"%init_type)
    if init_type=='normal':
        net.apply(weight_init_normal)
    elif init_type=='xavier':
         net.aplly(weight_init_xavier)
    elif init_type=='kaiming':
        net.apply(weight_init_kaiming)
    else :
        raise NotImplementedError("Initailsation Methos is not implemented[%s]"%init_type)




