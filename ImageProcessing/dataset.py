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
def rle2mask(rle: str, shape):
    """Convert RLE encoded string to binary mask."""
    width, height = shape
    mask = np.zeros(width * height, dtype=np.uint8)
    rle_numbers = [int(x) for x in rle.split()]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    for start, length in rle_pairs:
        mask[start - 1: start - 1 + length] = 1
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
    df=pd.read_csv(args.label_path)
    for file_name ,dft  in tqdm(df.groupby("ImageId")):
        if not (args.image_folder/file_name).exists():
            continue  
        height=dft.iloc[0]["Height"]
        width=dft.iloc[0]["Width"]
        size=Image.open(args.image_folder/file_name).size
        if (width,height)!=size:
              continue  
        mask=np.zeros((height,width),dtype=np.uint8)
        for i in dft.index:
            seg=dft.loc[i,"EncodedPixels"]
            mask=mask|rle2mask(seg,(width,height))
        if mask.sum()==0:
           continue
        shutil.copy(str(args.image_folder/file_name),str(output_image_folder/file_name))   
        cv2.imwrite(str(output_label_folder/f"{Path(file_name).stem}.png"),mask*255) 
if __name__=="__main__":
       main()         
          