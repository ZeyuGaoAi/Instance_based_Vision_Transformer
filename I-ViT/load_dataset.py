import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
)
import scipy.io as sio
from PIL import Image
import numpy as np
import heapq

       
    
    
class load_prcc_dataset_scnn_pos(Dataset):
    def __init__(self, txt_path, transform=None, train = True, N = 100,nuclues_size = 32,crop_path):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
       
        self.imgs = imgs       
        self.transform = transform
        self.train = train
        self.N = N
        self.nuclues_size = nuclues_size
        self.crop_path = crop_path

    def __getitem__(self,index):
        fn, label = self.imgs[index]
        slide_name = fn.split('/')[-2]
        patch_name = fn.split('/')[-1].strip('.png')
        
        crop_path = self.crop_path + '/' + str(self.nuclues_size) + '/' + slide_name+ '/' + patch_name + '.mat'
        crop = sio.loadmat(crop_path)
        nuclues = np.array(crop['nucleus'])
        cls_all = np.array(crop['class_all'])[0]
        
        cls_all=-cls_all
        cls_all.sort()
        cls_all = -cls_all
        cls_all = cls_all[:self.N]
        #print(len(cls_all))
        cls_list = [0 for i in range(self.N)]
        cls_list[:len(cls_all)] = cls_all
        
        nuclues = nuclues[:self.N]
        patches = []
        for i in nuclues :
            #temp =self.transform(self.train)(i)
            patches.append(i)
        

        pos = np.array(crop['cls_keep_xy'])
        pos = pos[:self.N]
        pos_list = [[0,0]for i in range(self.N)]
        pos_list[:len(pos)]=pos
        
        new_patch = np.zeros([3,self.nuclues_size,self.nuclues_size])
        while len(patches)<self.N:
            patches.append(new_patch)

        patches = torch.from_numpy(np.asarray(patches))

        cls_list = torch.as_tensor(np.asarray(cls_list), dtype=torch.int64)
        pos_list = torch.as_tensor(np.asarray(pos_list), dtype=torch.int64)
        return patches, cls_list, pos_list, label
        

    def __len__(self):
        return len(self.imgs)
    
    
    
