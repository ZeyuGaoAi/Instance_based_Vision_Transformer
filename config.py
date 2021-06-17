

import importlib
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorpack import imgaug

#### 
class Config(object):
    def __init__(self, ):

        self.seed = 10 
        mode = 'micronet'
        self.model_type = 'micronet'

        
        self.type_boundray = False # whether to predict the nuclear boundray
        self.type_nuclei = True
        self.type_classification = True # whether to predict the nuclear type
        # ! must use CoNSeP dataset, where nuclear type labels are available
        # denotes number of classes for nuclear type classification, 
        # plus the background class
        self.nr_types = 5
        # ! some semantic segmentation network like micronet,
        # ! nr_types will replace nr_classes if type_classification=True
        self.nr_classes = 2 # Nuclei Pixels vs Background
        
        self.auxilary_tasks = False # whether to use deep supervision or auxilary tasks strategy
        self.regression = True # whether to use np-regression branch
        self.gcb = True # whether to use attention block on encoder
        self.uncertainty = False # whether to use uncertainty loss weight strategy
        self.mix_class = True # single or multiple classification branches
        self.use_dice = False # whether to use dice loss

        # define your nuclei type name here, please ensure it contains
        # same the amount as defined in `self.nr_types` . ID 0 is preserved
        # for background so please don't use it as ID
        self.nuclei_type_dict = {
            'Grade1': 1, # ! Please ensure the matching ID is unique
            'Grade2': 2,
            'Grade3' : 3,
            'Endothelium': 4,
        }
        assert len(self.nuclei_type_dict.values()) == self.nr_types - 1

        #### Dynamically setting the config file into variable
        config_file = importlib.import_module('other') # fcn8, dcan, etc.
        config_dict = config_file.__getattribute__(self.model_type)

        for variable, value in config_dict.items():
            self.__setattr__(variable, value)
        #### Training data

        self.data_ext = '.npy' 

        # number of processes for parallel processing input
        self.nr_procs_train = 8 
        self.nr_procs_valid = 4 

        self.input_norm  = True # normalize RGB to 0-1 range

        ####
        self.save_dir = './%s' % self.model_type

        # path to checkpoints will be used for inference, replace accordingly
        self.inf_model_path  = self.save_dir + '/model-36000.index'#'/model-27000.index'

        # output will have channel ordering as [Nuclei Type][Nuclei Pixels][Additional]
        # where [Nuclei Type] will be used for getting the type of each instance
        # while [Nuclei Pixels][Additional] will be used for extracting instances
        # for inference during evalutaion mode i.e run by infer.py
        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']
        # for inference during training mode i.e run by trainer.py
        self.train_inf_output_tensor_names = ['predmap-coded', 'truemap-coded']

        self.inf_imgs_ext = '.png'

    def get_model(self):
        model_constructor = importlib.import_module('%s' % self.model_type)
        model_constructor = model_constructor.Graph       
        return model_constructor # NOTE return alias, not object

