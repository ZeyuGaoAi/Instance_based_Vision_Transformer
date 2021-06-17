# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:25:02 2020

@author: ZeyuGao
"""

import math
import os
from collections import deque

import cv2
import numpy as np

from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader

from config import Config

import json
import operator

from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
                                    binary_erosion,
                                    binary_dilation, 
                                    binary_fill_holes,
                                    distance_transform_cdt,
                                    distance_transform_edt)
from skimage.morphology import remove_small_objects, watershed
from postprocessing import *

def process(pred, model_mode, ws=True):
    def gen_inst_dst_map(ann):  
        shape = ann.shape[:2] # HW
        nuc_list = list(np.unique(ann))
        nuc_list.remove(0) # 0 is background

        canvas = np.zeros(shape, dtype=np.uint8)
        for nuc_id in nuc_list:
            nuc_map   = np.copy(ann == nuc_id)    
            nuc_dst = distance_transform_edt(nuc_map)
            nuc_dst = 255 * (nuc_dst / np.amax(nuc_dst))       
            canvas += nuc_dst.astype('uint8')
        return canvas
    
    if model_mode != 'dcan':
        assert len(pred.shape) == 2, 'Prediction shape is not HW'
        pred[pred  > 0.5] = 1
        pred[pred <= 0.5] = 0

        # ! refactor these
        ws = False if model_mode == 'unet' or model_mode == 'micronet' else ws
        if ws:
            dist = measurements.label(pred)[0]
            dist = gen_inst_dst_map(dist)
            marker = np.copy(dist)
            marker[marker <= 125] = 0
            marker[marker  > 125] = 1
            marker = binary_fill_holes(marker) 
            marker = binary_erosion(marker, iterations=1)
            marker = measurements.label(marker)[0]

            marker = remove_small_objects(marker, min_size=10)
            pred = watershed(-dist, marker, mask=pred)
            pred = remove_small_objects(pred, min_size=10)
        else:
            pred = binary_fill_holes(pred) 
            pred = measurements.label(pred)[0]
            pred = remove_small_objects(pred, min_size=10)
        
        if model_mode == 'micronet':
            # * dilate with same kernel size used for erosion during training
            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], np.uint8)
    
            canvas = np.zeros([pred.shape[0], pred.shape[1]])
            for inst_id in range(1, np.max(pred)+1):
                inst_map = np.array(pred == inst_id, dtype=np.uint8)
                inst_map = cv2.dilate(inst_map, kernel, iterations=1)
                inst_map = binary_fill_holes(inst_map)
                canvas[inst_map > 0] = inst_id
            pred = canvas
    else:
        assert (pred.shape[2]) == 2, 'Prediction should have contour and blb'
        blb = pred[...,0]
        blb = np.squeeze(blb)
        cnt = pred[...,1]
        cnt = np.squeeze(cnt)

        pred = blb - cnt # NOTE
        pred[pred  > 0.3] = 1 # Kumar 0.3, UHCW 0.3
        pred[pred <= 0.3] = 0 # CPM2017 0.1
        pred = measurements.label(pred)[0]
        pred = remove_small_objects(pred, min_size=20)
        canvas = np.zeros([pred.shape[0], pred.shape[1]])

        k_disk = np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], np.uint8)
        for inst_id in range(1, np.max(pred)+1):
            inst_map = np.array(pred == inst_id, dtype=np.uint8)
            inst_map = cv2.dilate(inst_map, k_disk, iterations=1)
            inst_map = binary_fill_holes(inst_map)
            canvas[inst_map > 0] = inst_id
        pred = canvas
        
    return pred

####
def process_instance_micro(pred_map, nr_types=0, output_dtype='uint16'):
    pred_inst = pred_map[...,0].copy()
    pred_inst = process(1-pred_inst, 'micronet')

    if nr_types != 0:
        pred_type = pred_map[..., 1:nr_types]
        pred_type = np.argmax(pred_type, axis=-1)
        pred_type = pred_type+1
        pred_type = np.squeeze(pred_type)

        pred_type_out = np.zeros([pred_type.shape[0], pred_type.shape[1]])               
        #### * Get class of each instance id, stored at index id-1
        pred_id_list = list(np.unique(pred_inst))[1:] # exclude background ID
        pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
        for idx, inst_id in enumerate(pred_id_list):
            inst_tmp = pred_inst == inst_id
            inst_type = pred_type[pred_inst == inst_id]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0: # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            pred_type_out += (inst_tmp * inst_type)
    else:
        pred_type_out = pred_inst.copy()
        pred_type_out[pred_type_out!=0] = 1

    return pred_inst.astype(output_dtype), pred_type_out.astype(output_dtype)

def process_instance_hcnet(pred_map, nr_types=0, output_dtype='uint16'):
    pos_map = pred_map[...,-1]*255
    binary_map = pred_map[...,-2]*255
    pred_inst = gaussianmap2binary(pos_map.astype('uint8'), binary_map.astype('uint8'), 150, 40)
    
    if nr_types != 0:
        pred_type = pred_map[..., 1:nr_types]
        pred_type = np.argmax(pred_type, axis=-1)
        pred_type = pred_type + 1
        pred_type = np.squeeze(pred_type)

        pred_type_out = np.zeros([pred_type.shape[0], pred_type.shape[1]])               
        #### * Get class of each instance id, stored at index id-1
        pred_id_list = list(np.unique(pred_inst))[1:] # exclude background ID
        pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
        for idx, inst_id in enumerate(pred_id_list):
            inst_tmp = pred_inst == inst_id
            inst_type = pred_type[pred_inst == inst_id]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0: # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            pred_type_out += (inst_tmp * inst_type)
    else:
        pred_type_out = pred_inst.copy()
        pred_type_out[pred_type_out!=0] = 1

    return pred_inst.astype(output_dtype), pred_type_out.astype(output_dtype)

####
class Inferer(Config):

    def __gen_prediction(self, x, predictor):
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x : input image to be segmented. It will be split into patches
                to run the prediction upon before being assembled back            
        """    
        step_size = self.infer_avalible_shape
        msk_size = self.infer_avalible_shape
        win_size = self.infer_input_shape
        crop_size = int((self.infer_mask_shape[0] - self.infer_avalible_shape[0])/2)

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)
        
        im_h = x.shape[0] 
        im_w = x.shape[1]

        last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), 'reflect')

        #### TODO: optimize this
        sub_patches = []
        # generating subpatches from orginal
        for row in range(0, last_h, step_size[0]):
            for col in range (0, last_w, step_size[1]):
                win = x[row:row+win_size[0], 
                        col:col+win_size[1]]
                sub_patches.append(win)

        pred_map = deque()
        while len(sub_patches) > self.inf_batch_size:
            mini_batch  = sub_patches[:self.inf_batch_size]
            sub_patches = sub_patches[self.inf_batch_size:]
            mini_output = predictor(mini_batch)[0]
#             print(mini_output.shape)
            mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = predictor(sub_patches)[0]
#             print(mini_output.shape)
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)
        
        #### Assemble back into full image
        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        #### Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        
        #### for crop_size !=0\
        if crop_size != 0:
            pred_maps = []
            for i in range(pred_map.shape[0]):
                one_pred_map = pred_map[i][crop_size:-crop_size,crop_size:-crop_size,...]
                pred_maps.append(one_pred_map)
            pred_maps = np.array(pred_maps)
            pred_map = pred_maps
        
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                        np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1], 
                                         pred_map.shape[2] * pred_map.shape[3], ch))
        pred_map = np.squeeze(pred_map[:im_h,:im_w]) # just crop back to original size
        return pred_map

    ####
    def run(self, file_path):

        model_path = self.inf_model_path

        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model        = model_constructor(),
            session_init = get_model_loader(model_path),
            input_names  = self.eval_inf_input_tensor_names,
            output_names = self.eval_inf_output_tensor_names)
        predictor = OfflinePredictor(pred_config)
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred_map = self.__gen_prediction(img, predictor)
#         for i in file_path:
#             img = cv2.imread(i)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             pred_map = self.__gen_prediction(img, predictor)
        return pred_map
    
