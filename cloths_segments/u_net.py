import os 
from torch import nn
import  segmentation_models_pytorch as smp
import matplotlib.pyplot as plt 
from dotenv import load_dotenv
from collections import namedtuple
from torch.utils import model_zoo
from iglovikov_helper_functions.dl.pytorch.utils import rename_layers
from segmentation_models_pytorch import Unet
load_dotenv()
model=namedtuple("model",["url","factory"])
#store a factory, not an instance -> every call to create_model() gets its own weights
models={
    "Unet_2020-10-30":model(
        url="https://github.com/ternaus/cloths_segmentation/releases/download/0.0.1/weights.zip",
        factory=lambda: Unet(encoder_name="timm-efficientnet-b3",classes=1,encoder_weights=None)
    )
}
def create_model(model_name:str)->nn.Module:
     entry=models[model_name]
     model=entry.factory()
     state_dict=model_zoo.load_url(entry.url,progress=True,map_location="cpu")["state_dict"]
     state_dict=rename_layers(state_dict,{"model.":""})
     model.load_state_dict(state_dict)
     return model 
