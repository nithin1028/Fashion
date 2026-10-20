from torch.utils.tensorboard.summary import hparams
from pytorch_lightning.trainer.connectors.logger_connector import result
from cv2 import sepFilter2D
import argparse
import os 
from pathlib import Path
from typing import Dict
import pytorch_lightning as pl 
import torch
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from iglovikov_helper_functions.dl.pytorch.utils import state_dict_from_disk
from pytorch_lightning.loggers import WandbLogger
from pytorch_toolbelt.losses import JaccardLoss,BinaryFocalLoss
from torch.utils.data import DataLoader
from cloths_segments.data_process import SegmenatationDataset
from cloths_segments.metrics_score import binary_mean_union_intersection
from cloths_segments.utils import get_samples

image_path=Path(os.environ["IMAGE_PATH"])
mask_path=Path(os.environ["MASK_PATH"])
def get_args():
    parser=argparse.ArgumentParser()
    arg=parser.add_argument
    arg("-c","--config_path",type=Path,help="Path tp the config",required=True)
    return parser.parse_args()
class SegmentPeople(pl.LightningModule):
    def __init__(self,hparams):
        super().__init__()
        self.hparams=hparams 
        self.model=object_from_dict(self.hparams["model"])
        if "resume_form_checkpoint" in self.hparams:
            correlation:Dict[str,str]={"model.":""}
            state_dict=state_dict_from_disk(
                   file_path=self.hparams["resume_form_checkpoint"],
                   rename_in_layers=correlation
            )   
            self.model.load_state_dict(state_dict)
            self.losses=[
                   ("jaccard",0.1,JaccardLoss(mode="binary",from_logits=True)),
                   ("focal",0.9,BinaryFocalLoss())
            ]
    def forward(self,batch:torch.Tensor)->torch.Tensor:
        return self.model(batch)  
    def setup(self,stage=0):
        samples=get_samples(image_path,mask_path)
        num_train=int((1-self.hparams["val_split"])*len(samples)) 
        self.train_samples=samples[:num_train]
        self.val_samples=samples[num_train]
        print("Length train_samples=",len(self.train_samples))
        print("Length of val_samples=",len(self.val_samples))                 
    def train_dataloader(self):
        train_aug=from_dict(self.hparams["train_aug"])
        if "epoch_length" not in self.hparams["train_parameters"]:
            epoch_length=None
        else :
            epoch_length=self.hparams["train_parameters"]["epoch_length"]
        result=DataLoader(
                SegmenatationDataset(self.train_sample,train_aug,epoch_length) ,
                batch_size=self.hparams["train_parameters"]["batch_size"] ,
                num_workers=self.hparams["num_workers"],
                shuffle=True,
                pin_memory=True,
                drop_last=True

            )   
        print("Train Dataloder:",len(result))
        return result
    def val_dataloader(self):
        val_agu=from_dict(self.hparams["train_agu"])
        result=DataLoader(
               SegmenatationDataset(self.val_samples,val_agu,lenght=None),
               batch_sampler=self.hparams["val_parameters"]["batch_size"],
               shuffle=True,
               pin_memory=True,
               drop_last=True
        ) 
        print("Train dataloder=",len(result))
        return result  
    def configure_optimizer(self):
        optimiser=object_from_dict(
            self.hparams["optimiser"],
            params=[x for x in self.model.parameters() if x.requires_grad]
        
        ) 
        scheduler=object_from_dict(
                self.hparams["scheduler"],
                optimiser=optimiser
        )
        self.optimiser=[optimiser]
        return self.optimiser,[scheduler]
    def training_step(self,batch,bathc_idx):
        features=batch["features"]
        masks=batch["mask"]
        logits=self.forward(features) 
        total_loss=0
        logs={}
        for loss_name,weight,loss in self.losses:
            ls_mask=loss(logits,masks)
            total_loss+=weight*ls_mask
            logs[f"train_loss_{loss_name}"]=total_loss
        logs["train_loss"]=total_loss
        logs["lr"]=self._get_current_lr()
        return {"loss":total_loss,"log":logs}
    def _get_current_lr(self):
        lr=[x["lr"] for x in self.optimiser[0].param_groups][0]
        return torch.Tensor([lr])[0].cuda()
    def validation_step(self,batch,batch_id):
        features=batch["features"]
        mask=batch["mask"]
        logits=self.forward(features)
        result={}
        for loss_name,_,loss in self.losses:
            result[f"val_mask_{loss_name}"]=loss(logits,mask)
        result["val_iou"]=binary_mean_union_intersection(logits,mask)    
        return result
    def validation_epoch_end(self,outputs):
        logs={"epoch":self.trainer.current_epoch}  
        avg_val_iou=find_average(outputs,"val_iou")
        logs["val_iou"]=avg_val_iou
        return {"val_iou":avg_val_iou,"log":logs}
    def main():
        args=get_args()
        with open(args.config_path) as f :
            hparams=yaml.load(f,Loader=yaml.SafeLoader)
        pipeline=SegmentPeople(hparams)
        Path(hparams["checkpoint_callback"]["filepath"]).mkdir(exist_ok=True,parents=True)
        trainer =object_from_dict(
                 hparams["trainer"],
                 logger=WandbLogger(hparams["experiment_name"]),
                 checkpoint_callback=object_from_dict(
                      hparams["checkpoint_callback"]
                 )
                 
        )
        trainer.fit(pipeline)
    if __name__=="__main__":
        main()    

        


     