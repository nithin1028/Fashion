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
class FeatureExtraction(nn.Module):
    def __init__(self,input_nc=3,out_channels=64,n_layers=3,norm_layers=nn.BatchNorm2d,use_dropout=False):
          super(FeatureExtraction,self).__init__()
          model=[nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1),nn.ReLU(inplace=True),norm_layers(out_channels)]
          for i in range(n_layers):
            in_ngf=2**i*out_channels if 2**i*out_channels<512 else 512 
            out_ngf=2**i**out_channels if 2**i*out_channels else 512
            model+=[nn.Conv2d(in_ngf,out_ngf,kernel_size=4,stride=2,padding=1),nn.ReLU(inplace=True)]
            model+=[norm_layers(out_ngf)]
            model+=[nn.Conv2d(512,512,kernel_size=4,stride=2,padding=1),nn.ReLU(inplace=True)]
            model+=[norm_layers(512)]
            model+=[nn.Conv2d(512,512,kernel_size=4,stride=2,padding=1)]
            self.model=nn.Sequential(*model)
            init_weight(self.model,init_type='normal')
    def forward(self,x):
        return self.model(x)



