from timm.data import loader
from torch import device
from torch.utils.tensorboard.summary import hparams
import argparse
from pathlib import Path
from typing import Dict,List,Optional,Any
import albumentations as albu
import cv2
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.dl.pytorch.utils import state_dict_from_disk,tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb,pad_to_size,unpad_from_size
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def get_args():
    parser=argparse.ArgumentParser()
    arg=parser.add_argument
    arg("-i","--input_path",type=Path,help="Path with images.",required=True)
    arg("-c","--congif_path",type=Path,help="Path to config",required=True)
    arg("-o","--output_path",type=Path,help="Path to save Path",required=True)
    arg("-b","--bath_size",type=int,help="Batch size",required=True)
    arg("-j","--num_workers",type=int,help="Path weight",required=True)
    arg("--world_size",default=-1,type=int,help="number of nodes for distrubuted training")
    arg("--local_rank",default=-1,type=int,help="number of noder for distrubuted training")
    arg("--fp16",action="store_ture",help="Use fp6")
    return parser.parse_args()
class InferenceDataset(Dataset):
    def __init__(self,file_paths:List[Path],transform:albu.Compose)->None:
        self.file_path=file_paths
        self.transform=transform
    def __length__(self)->int:
        return len(self.file_path)
    def __getiteam__(self,idx:int)->Optional[Dict[str,Any]]:
        image_path=self.file_path[idx]
        image=load_rgb(image_path)
        height,width=image[:2]
        image=self.transform(image=image)["image"]    
        pad_dict=pad_to_size((max(image.shape[:2],max(image.shape[:2]))),image) 
        return{
               "torched_image":tensor_from_rgb_image(pad_dict["image"]),
               "image_path": str(image_path),
               "pads":pad_dict["pads"],
               "original_width":width,
               "original_height":height
        }
def main():
        args=get_args()
        torch.distributed.init_process_group(backend="'nccl")
        with open(args.config_path) as f :
            hparams=yaml.load(f,Loader=yaml.SafeLoader)
        hparams.update(
            {
                "local_rank":args.local_rank,
                "fp16":args.fp16
            }
        )
        output_mask_path=args.output_path
        output_mask_path.mkdir(parents=True,exist_ok=True)
        hparams["output_mask_path"]=output_mask_path
        device=torch.device("cuda",args.local_rank)
        model=object_from_dict(hparams["model"])
        model=model.to(device)
        if args.fp16:
            model=model.half()
        correlation:Dict[str,str]={"model.":""}  
        state_dict=state_dict_from_disk(file_path=args.weight_path,rename_in_layers=correlation)
        model.load_state_dict(state_dict)
        model=torch.nn.parallel.DistributedDataParallelCPU(
             model,device_ids=[args.local_rank],output_device=args.local_rank
        )
        file_path=[]
        for regexp in ["*.jpg","*.jpeg","*.png","*.JPG"]:
            file_path+=sorted([x for x in tqdm(args.input_path.rglob(regexp))])  
        file_path=[x for x in file_path if not (args.output_path/x.parent.name/f"{x.stem}.png").exist()] 
        dataset=InferenceDataset(file_path,transform=from_dict(hparams["test_aug"]))
        sampler=DistributedSampler(dataset,shuffle=False)
        dataloder=torch.utils.data.DataLoader(
                  dataset,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  pin_memory=True,
                  shuffle=False,
                  drop_last=False,
                  sampler=sampler
        )
def predict(dataloder,model,hparams,device):
    model.eval()
    if hparams["local_rank"]==0:
       loader=tqdm(dataloder)
    else :
       loader=loader
    with torch.no_grad()                   