from pathlib import Path
from typing import List,Dict,Any,Tuple
import albumentations as albu 
from iglovikov_helper_functions.utils.image_utils import load_grayscale,load_rgb
from pytorch_toolbelt.utils.torch_utils import image_to_tensor
from torch.utils.data import Dataset
import numpy as np
import torch
class SegmenatationDataset(Dataset):
    def __init__(self,samples:List[Tuple[Path,Path]],transform:albu.Compose,length:int=None)->None:
        self.samples=samples
        self.transform=transform
        if length is None:
            self.length=len(self.samples)
        else:
            self.length=length
    def __len__(self)->int:
        return self.length
    def __getitem__(self,idx:int)->Dict[str,Any]:
        #one corrupt/unreadable image must not kill the whole epoch
        last_error=None
        for _ in range(min(10,len(self.samples))):
            i=idx%len(self.samples)
            idx+=1
            image_path,mask_path=self.samples[i]
            try:
                image=load_rgb(image_path,lib="cv2")
                mask=load_grayscale(mask_path)
                sample=self.transform(image=image,mask=mask)
                image,mask=sample["image"],sample["mask"]
                mask= (mask>0).astype(np.uint8) 
                mask=torch.from_numpy(mask)
                return{
                    "image_id":image_path.stem,
                    "features":image_to_tensor(image),
                    "mask":torch.unsqueeze(mask,0).float()  
                }        
            except Exception as error:
                last_error=error
                continue
        raise RuntimeError(f"could not load a valid sample near index {idx}") from last_error
