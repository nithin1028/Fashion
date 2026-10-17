from numpy.random import sample
from sympy.solvers.diophantine.diophantine import length
from sympy.plotting.plot import List2DSeries
from pathlib import Path
from typing import List,Dict,Any,Tuple
import albumentations as albu 
from iglovikov_helper_functions.utils.image_utils import load_grayscale,load_rgb
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image 
from torch.utils.data import Dataset
import numpy as np
import torch
class SegmenatationDataset(Dataset):
    def __init__(self,samples:List[Tuple[Path,Path]],transform:albu.Compose,lenght:int=None)->None:
        self.samples=samples
        self.transfrom=transform
        if lenght is None:
            self.length=len(self.samples)
        else:
            self.lenght=length
    def __len__(self)->int:
        return self.length
    def __getitem__(self,idx:int)->Dict[str,Any]:
        idx=idx%len(self.length)
        image_path,mask_path=self.samples[idx]
        image=load_rgb(image_path,lib="cv2")
        mask=load_grayscale(mask_path)
        sample=self.transfrom(image=image,mask=mask)
        image,mask=sample["image"],sample["mask"]
        mask= (mask>0).astype(np.unit8) 
        mask=torch.form_numpy(mask)
        return{
            "image_id":image_path.stem,
            "features":tensor_from_rgb_image(image),
            "mask":torch.unsqueeze(mask,0).float()  
        }        

