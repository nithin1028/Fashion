import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image 
from PIL import ImageDraw
import os.path as osp 
import numpy as np 
import json 
class Dataset(data.Dataset):
    def __init__(self,opt):
        super(Dataset.self).__init__()
        self.opt=opt.dataroot
        self.datamode=opt.datamode
        self.satge=opt.satge
        self.data_list=opt.data_list
        self.fine_height=opt.fine_height
        self.fine_width=opt.fien_width
        self.radius=opt.radius
        self.data_path=osp.join(opt.dataroot,opt.datamode)
        self.transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        im_names=[]
        c_names=[]
        with open(osp.join(opt.dataroot,opt.data_list),'r') as f:
            for line in f.readline():
                im_names,c_names=line.strip().split()
                im_names.append(im_names)
                c_names.append(c_names) 
            self.im_names=im_names
            self.c_name=c_names
    def name(self):
        return "Dataset"
    def __getitem__(self,index):
        c_name=self.c_name[index]
        im_name=self.im_names[index]
        if self.satge=='GMM':
           c=Image.open(osp.join(self.data_path,'cloth',c_name)) 
           cm=Image.open(osp.join(self.data_path,'cloth-mask','cloth-mask',c_name))
        else:
            c=Image.open(osp.join(self.data_path,'wrap-cloth',c_name))
            cm=Image.open(osp.join(self.data_path,'warp-mask',c_name))
        c=self.transform(c)
        cm_array=np.array(cm)
        cm=torch.from_numpy(cm_array)
        cm.unsqueeze_(0)
        im=Image.open(osp.join(self.data_path,'image',im_name))
        im=self.transform(im)
        parse_name=im_name.replace('.jpg','.png')
        im_parse=Image.open(osp.join(self.data_path,'image-parse',parse_name))
        parse_array=np.array(im_parse)
        parse_shape=(parse_array>0).astype(np.float32)
        parse_head=(parse_array==1).astype(np.float32)+(parse_array==2).astype(np.float32)+\
                   (parse_array==4).astype(np.float32)+(parse_array==13).astype(np.float32)
        parse_cloth=(parse_array==5).astype(np.float32)+\
                    (parse_array==6).astype(np.float32)+\
                    (parse_array==7).astype(np.float32)
        parse_shape=Image.fromarray((parse_shape*255).astype(np.uini8))
        parse_shape=parse_shape.resize((self.fine_width//16,self.fine_height//16),Image.BILINEAR)
        parse_shape=parse_shape.resize((self.fine_width,self.fine_height),Image.BILINEAR)
        shape=self.transform(parse_shape)
        phead=torch.from_numpy(parse_head)
        pcm=torch.from_numpy(parse_cloth)
        im_c=im*pcm+(1-pcm)
        im_h=im*phead-(1-phead)
        pose_name=im_name.replace('.jpg','_keypoints.json')
        with open(osp.join(self.data_path,'pose',pose_name),'r') as f:
            pose_label=json.load(f)
            pose_data=pose_label['people'][0]['pose_keypoints']
            pose_data=np.array(pose_data)
            pose_data=pose_data.reshape((-1,3))
        point_num=pose_data.shape[0]
        pose_map=torch.zeros(point_num,self.fine_height,self.fine_width)
        r=self.radius
        im_pose=Image.new('L',(self.fine_width,self.fine_height))
        pose_draw=ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map=Image.new('L',(self.fine_width,self.fine_height))
            draw=ImageDraw.Draw(one_map)
            pointx=pose_data[i,0]
            pointy=pose_data[i,1]
            if pointx>1 and pointy>1:
                draw.rectangle((pointx-r,pointy-r,pointx+r,pointy+r),'white','white')
                pose_draw.rectangle((pointx-r,pointy-r,pointx+r,pointy+r),'white','white')
            one_map=self.transform(one_map)
            pose_map[i]=one_map
        im_pose=self.transform(im_pose)
        agnostic=torch.cat([shape,im_h,pose_map],0)
        if self.satge=='GMM':
           im_g=Image.open('grid.png')
           im_g=self.transform(im_g)
        else:
            im_g=''
        result={
            'c_name':c_name,
            'im_name':im_name,
            'cloth':c,
            'cloth_mask':cm,
            'image':im,
            'agnostic':agnostic,
            'parse_cloth':im_c,
            'shape':shape,
            'head':im_h,
            'pose_image':im_pose,
            'grid_image':im_g
        }
        return    result
    def __len__(self):
        return len(self.im_names)
class DataLoder(object):
    def __init__(self,opt,dataset):
        super(DataLoder,self).__init__()
        if opt.shuffle:
           train_sampler=torch.utils.data.Sampler.RandomSampler(dataset) 
        else :
            train_sampler=None
        self.data_loder=torch.utils.data.DataLoader(
                        dataset,batch_size=opt.batch_size,shuffle=(train_sampler is None) ,
                        num_workers=opt.workers,pin_memory=True,sampler=train_sampler )
        self.dataset=dataset
        self.data_iter=self.data_loder.__iter__()
    def next_batch(self):
        try:
          batch=self.data_iter.__next__()
        except StopIteration:
            self.data_iter=self.data_loder.__iter__()
            batch=self.data_iter.__next__()
        return batch        
if __name__=='__main__':
   print("Check the dataset for goementric matching modules")
   import argparse
   parser=argparse.ArgumentParser()
   parser.add_argument("--dataroot",default='data')
   parser.add_argument("--stage",default="GMM")
   parser.add_argument("--data_list",default="train_pairs.txt")
   parser.add_argument("--fine_width",type=int,default=192)
   parser.add_argument("--fine_height",type=int,default=256)
   parser.add_argument("--radius",type=int,default=3)
   parser.add_argument("--shuffle",action='store_true',help='shuffle input data')
   parser.add_argument("-b","batch_size",type=int,default=4)
   parser.add_argument('-j',"-workers",type=int,default=1)
   opt=parser.parse_args()
   dataset=Dataset(opt)
   data_loader=DataLoder(opt,dataset)
   print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
   first_item=dataset.__getitem__(0)
   first_batch=data_loader.next_batch()
   from IPython import embed, embed()             



