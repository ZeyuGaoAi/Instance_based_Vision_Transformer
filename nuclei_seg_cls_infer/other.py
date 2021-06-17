import numpy as np
import tensorflow as tf
from misc import * 

micronet = {
    'train_input_shape' : [252, 252],
    'train_mask_shape'  : [252, 252],
    'infer_input_shape' : [252, 252],
    'infer_mask_shape'  : [252, 252], 
    'infer_avalible_shape': [200, 200],

    'training_phase'    : [
        {
            'nr_epochs': 80,
            'manual_parameters' : {
                'learning_rate': (1.0e-4, [('40', 1.0e-5)]),
                'aux_loss_dw'  : (1.0, 
                        [(str(epoch), 1.0 / epoch) for epoch in range(2, 251)]
                    ),
            },
            'train_batch_size'  : 8,
            'infer_batch_size'  : 16,

            'model_flags' : {
            }
        }
    ],

    'optimizer'         : tf.train.AdamOptimizer,

    'inf_batch_size'  : 16,
    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
}

hcnet = {
    'train_input_shape' : [262, 262],
    'train_mask_shape'  : [256, 256],
    'infer_input_shape' : [262, 262],
    'infer_mask_shape'  : [256, 256], 
    'infer_avalible_shape': [200, 200],

    'training_phase'    : [

        {
            'nr_epochs': 50,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('25', 1.0e-5)]), 
            },
            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained_path'  : '../ImageNet-ResNet34.npz',
            'train_batch_size' : 4, # unfreezing everything will
            'infer_batch_size' : 8,

            'model_flags' : {
                'freeze' : True,
            }
        },
        {
            'nr_epochs': 50,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('25', 1.0e-5)]), 
            },
            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained_path'  : -1,
#             'pretrained_path'  : '../ImageNet-ResNet34.npz',
            'train_batch_size' : 3, # unfreezing everything will
            'infer_batch_size' : 6,

            'model_flags' : {
                'freeze' : False,
            }
        },
    ],

    'optimizer'           : tf.train.AdamOptimizer,

    'inf_batch_size'    : 16,
    'inf_auto_metric'   : 'valid_mean_dice',
    'inf_auto_comparator' : '>',
}