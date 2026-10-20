from pylab import imshow
from u_net import create_model
from iglovikov_helper_functions.utils.image_utils import load_rgb,unpad,pad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
import numpy as np
import albumentations as albu
import torch
import cv2
import os 
model=create_model("Unet_2020-10-30")
model.eval()
image=load_rgb(os.path.join(os.path.dirname(__file__),"Bg.jpeg"))
imshow(image)
transform=albu.Compose([albu.Normalize(p=1)],p=1)
padded_image,pad=pad(image,factor=32,border=cv2.BORDER_CONSTANT)
x=transform(image=padded_image)["image"]
x=torch.unsqueeze(tensor_from_rgb_image(x),0)
with torch.no_grad():
     prediction=model(x)[0][0]
mask=(prediction>0).cpu().numpy().astype(np.uint8)
mask=unpad(mask,pad)    
imshow(mask)
dst=cv2.addWeighted(image,1,(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)*(0,255,0)).astype(np.uint8),0.5,0)
imshow(dst)
imshow(np.hstack([image,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)*255,dst]))
