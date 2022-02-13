import os
import os.path
import sys

import cv2
import numpy as np
import  matplotlib.pyplot as plt
import torch
from PIL import Image

from torch.utils.data import DataLoader,Dataset
import h5py
from Dataset import transforms

iheight, iwidth = 480, 640 # raw image size
def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth

class NYUDataset(Dataset):

    def is_h5file(self,filename):
        return (filename.endswith('.h5'))


    def getdata(self,path):
        img_paths=[]
        dirs = os.listdir(path)
        for dir in dirs:
            subpath=os.path.join(path,dir)
            for root,dir_name_list,file_list in os.walk(subpath):

                for file in file_list:
                    if self.is_h5file(file):
                        file_path=os.path.join(root,file)
                        img_paths.append(file_path)
        return img_paths

    def train_transform(self, rgb, depth):

        do_flip = np.random.uniform(0.0, 1.0) > 0.5  # random horizontal flip
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        depth_np = depth
        transform = transforms.Compose([

            transforms.Rotate(angle),
            transforms.Resize((240,320)),
           transforms.CenterCrop(self.output_size),
           transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        depth_np = transform(depth_np)
        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize((240,320)),
            transforms.CenterCrop(self.output_size)
        ])

        rgb_np = transform(rgb)
        depth_np = transform(depth_np)

        return rgb_np, depth_np
    def __init__(self,path,split):
        self.output_size=(228,304)
        self.max_depth=10.0
        self.split=split

        self.imgs_path=self.getdata(path)
        self.loader=h5_loader
        if split=='train':
            self.transform=self.train_transform
        elif split=='val':
            self.transform=self.val_transform
        else:
            raise (RuntimeError("Invalid dataset split: " + split + "\n"
                                                                    "Supported dataset splits are: train, val"))


    def __getitem__(self, index):

        rgb,depth=self.loader(self.imgs_path[index])

        rgb,depth=self.transform(rgb,depth)
        totensor=transforms.ToTensor()
        input_tensor=totensor(rgb)/255   #(3,228,304)
        dpt=totensor(depth)
        dpt_tensor=dpt.unsqueeze(0)     # (1,228,304)


        return input_tensor, dpt_tensor

    def __len__(self):
        return len(self.imgs_path)




