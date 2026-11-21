"""
Process the Dataset https://www.kaggle.com/competitions/imaterialist-fashion-2019-FGVC6/data?select=train.csv
"""
import argparse
import shutil
from pathlib import Path
import cv2 
import numpy as np 
import pandas as pd 
from PIL import  Image 
from tqdm import tqdm
def _fill_rle(flat: np.ndarray, rle) -> None:
    """Fill one RLE into a preallocated flat buffer (no per-garment allocation)."""
    if not isinstance(rle, str) or not rle:
        return  # NaN / empty EncodedPixels -> image with no annotation
    numbers = np.array(rle.split(), dtype=np.int64)
    if numbers.size < 2:
        return
    for start, length in numbers.reshape(-1, 2):
        flat[start - 1: start - 1 + length] = 1

def rle2mask(rle: str, shape):
    """Convert RLE encoded string to binary mask."""
    width, height = shape
    mask = np.zeros(width * height, dtype=np.uint8)
    _fill_rle(mask, rle)
    return mask.reshape((height, width), order='F')

def get_args():
    parse=argparse.ArgumentParser()
    arg=parse.add_argument
    arg("-i","--image_folder",type=Path,help="Path to folder with image")
    arg("-l","--label_path",type=Path,help="Path to cvs with lables")
    arg("-o","--output_folder",type=Path,help="Path to thr out folder")
    return parse.parse_args()
def main():
    args=get_args()
    output_image_folder=args.output_folder/"images"
    output_image_folder.mkdir(exist_ok=True,parents=True)
    output_label_folder=args.output_folder/"labels"
    output_label_folder.mkdir(exist_ok=True,parents=True)
    df=pd.read_csv(
        args.label_path,
        usecols=["ImageId", "EncodedPixels", "Height", "Width"],
        dtype={"ImageId": str, "EncodedPixels": str},
    )
    for file_name ,dft  in tqdm(df.groupby("ImageId", sort=False)):
        if not (args.image_folder/file_name).exists():
            continue  
        height=dft.iloc[0]["Height"]
        width=dft.iloc[0]["Width"]
        size=Image.open(args.image_folder/file_name).size
        if (width,height)!=size:
              continue  
        mask_flat=np.zeros(height*width,dtype=np.uint8)
        for seg in dft["EncodedPixels"].tolist():
            _fill_rle(mask_flat,seg)
        mask=mask_flat.reshape((height,width),order='F')
        if mask.sum()==0:
           continue
        shutil.copy(str(args.image_folder/file_name),str(output_image_folder/file_name))   
        cv2.imwrite(str(output_label_folder/f"{Path(file_name).stem}.png"),mask*255) 
if __name__=="__main__":
       main()         
          