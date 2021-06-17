import os
from PIL import Image
import pandas as pd
import numpy as np
import time, threading
from multiprocessing import Pool
import scipy.io as sio
import my_transform as T
import scipy.stats as st
import time
def transform():
#     base_size = args.img_size + 64
#     crop_size = args.img_size
    transforms = []
#     if train:
#         transforms.append(T.Resize(1000))
#         transforms.append(T.RandomHorizontalFlip(0.5))
#         transforms.append(T.RandomCrop(base_size, crop_size))
#         transforms.append(T.RandomRotation((-5,5)))
    transforms.append(T.ToTensor())
    #transforms.append(T.Normalize(mean=[0.681, 0.486, 0.630], std=[0.213, 0.256, 0.196]))
    return T.Compose2(transforms)

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
#     print(pred_id)
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1    
    return new_pred

def deal(crop_size,path,mask_path,out_path,js):
    print(js,path)
    fn = path
    slide_name = fn.split('/')[-2]
    patch_name = fn.split('/')[-1].strip('.png')
    img = Image.open(fn).convert('RGB')
    
    seg_path = mask_path + '/' + fn.split('/')[-2] + '/' + fn.split('/')[-1].strip('.png') + '.mat'
    instance_map = sio.loadmat(seg_path)['Instance']
    class_map = sio.loadmat(seg_path)['Type']
    instance_map[class_map == 4] = 0
    instance_map = remap_label(instance_map, by_size=True)
    #instance_map[instance_map > 255] = 0
    instance_map = Image.fromarray(instance_map.astype('uint32'))
    class_map = Image.fromarray(class_map.astype('uint32'))
    
    img, instance_map, class_map = transform()(img, instance_map, class_map)
    instance_map = remap_label(np.asarray(instance_map), by_size=True)
    class_map = np.asarray(class_map)
    
    cls_1_xy = []
    cls_1 = []
    cls_2_xy = []
    cls_2 = []
    cls_3_xy = []
    cls_3 = []
    for inst in range(len(np.unique(instance_map))-1):
        x = np.where(instance_map==(inst+1))[0]
        y = np.where(instance_map==(inst+1))[1]
        cls = class_map[instance_map==(inst+1)]
        cls = np.median(cls)
        if cls == 1:
            cls_1.append(inst+1)
            cls_1_xy.append([int((x.min()+x.max())/2),int((y.min()+y.max())/2)])
        if cls == 2:
            cls_2.append(inst+1)
            cls_2_xy.append([int((x.min()+x.max())/2),int((y.min()+y.max())/2)])
        if cls == 3:
            cls_3.append(inst+1)
            cls_3_xy.append([int((x.min()+x.max())/2),int((y.min()+y.max())/2)])

    cls_3_xy = [x for _, x in sorted(zip(cls_3, cls_3_xy), key=lambda pair: pair[0])]
    cls_3.sort()
    cls_2_xy = [x for _, x in sorted(zip(cls_2, cls_2_xy), key=lambda pair: pair[0])]
    cls_2.sort()
    cls_1_xy = [x for _, x in sorted(zip(cls_1, cls_1_xy), key=lambda pair: pair[0])]
    cls_1.sort()

    cls_keep = cls_3 + cls_2 + cls_1
    #cls_keep = cls_keep[:self.N]

    cls_keep_xy = cls_3_xy + cls_2_xy + cls_1_xy
    #cls_keep_xy = cls_keep_xy[:self.N]

    cls_all = [3 for i in range(len(cls_3))] + [2 for i in range(len(cls_2))] + [1 for i in range(len(cls_1))]

    #cls_all = cls_all[:self.N]

    

    cls_all = [x for _, x in sorted(zip(cls_keep, cls_all), key=lambda pair: pair[0])]
    cls_xy = [x for _, x in sorted(zip(cls_keep, cls_keep_xy), key=lambda pair: pair[0])]

    cls_keep.sort()

    
    img = np.asarray(img)
    img_size = img.shape[1]
    patches = []
    # crop nuclei images
    n = int(crop_size/2)

    for xy in cls_keep_xy:
        new_patch = np.zeros([3,crop_size,crop_size])
        
        patch = img[:, max(0,(xy[0]-n)):min((xy[0]+n),img_size), max(0,(xy[1]-n)):min((xy[1]+n),img_size)]
        #print(patch.shape)
        new_patch[:,:patch.shape[1], :patch.shape[2]] = patch
        patches.append(new_patch)

    nucleus = {'nucleus':patches,'class_all':cls_all,'cls_xy':cls_xy,'cls_keep_xy':cls_keep_xy}
    sio.savemat(out_path, nucleus)








if __name__ == '__main__':

    #dataset path
    path = '/home5/hby/PRCC/New_Data/dataset.txt'
    #nuclues segmentation and grading mask path
    mask_path = '/home5/gzy/PRCCDataset/Nuclei_Prediction_2000_new'
    outfiles = '/home5/hby/PRCC/New_Data/crop/'
    
    fh = open(path, 'r')
    imgs = []
    for line in fh:
        line = line.rstrip()
        words = line.split()
        imgs.append((words[0], int(words[1])))

    js=0
    #the nuclues crop size
    crop_size=32
    for i in imgs:
        js+=1
        print(js)
        slide_name = i[0].split('/')[-2]
        patch_name = i[0].split('/')[-1].strip('.png')
        out_file = outfiles + str(crop_size) + '/' + slide_name 
        if not os.path.exists(out_file):
            os.makedirs(out_file)
        out_path = out_file+'/'+patch_name+'.mat'
        if not os.path.exists(out_path):
            print(js,i)
            deal(crop_size,i[0],mask_path,out_path,js)


