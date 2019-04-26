from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.SAMPLE_ONE = False

# Wether to train the model based on lower resolution results
__C.NUM_D = 2 # Multiple discriminator using in pix2pixHD
__C.DIS_HD = False # Using pix2pixHD discriminator setting or not
__C.OUTPUT_DIR = '' # Output directory path
__C.CONFIG_NAME = ''
__C.GPU_ID = '0' # GPU ID 
__C.CUDA = True # Use GPU or not
__C.WORKERS = 4 # The CPU worker number in data input pipeline
__C.TOTAL_SIZE = 0 # The number of samples in dataset



"""
    Generator and Discriminator's hyperparameter setting:

    Although NGF and NDF are given, the hyperparameters of networks are 
        specified directly in the code using the default parameters given
        by baseline models
"""
#--------------------------------------------------
# Generator related setting
#--------------------------------------------------
__C.NET_G = '' # Generator's checkpoint name in 
__C.G_TYPE = '' # Generator type
__C.NGF = 64

#--------------------------------------------------
# Discriminator related setting
#--------------------------------------------------
__C.D_TYPE = '' # if '' then to Spectral Normalization and Projection
__C.NET_D = '' # Discriminator's checkpoint name
__C.NDF = 64


#--------------------------------------------------
# Dataset related setting
#--------------------------------------------------
__C.DATASET_NAME = 'shoes' # Dataset name
__C.DATA_DIR = '' # The dataset folder
__C.IMSIZE = 256
__C.IM_RATIO = 1 
__C.LOAD_SIZE = 280
__C.NO_CROP = False # Cropping in data input pipeline
__C.NO_FLIP = False # Flipping in data input pipeline
__C.Z_NUM = 256
__C.CLASS_NUM = 1 # Input condition's channel number

#--------------------------------------------------
# Training runtime setting
#--------------------------------------------------
__C.NO_DECAY = False # No learning rate decay
__C.DEBUG_FLAG = False # Debug mode
__C.L1_LOSS = False # Pixel-wise L1 regression loss
__C.VGG_LOSS = False # VGG LOSS 

# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True # Training or not
__C.TRAIN.BATCH_SIZE = 64 # Batch size for each GPU using
__C.TRAIN.MAX_EPOCH = 120 # Maximum trianing epoch number
__C.TRAIN.GENERATOR_LR = 1e-4 # Generator's initial learning rate
__C.TRAIN.DISCRIMINATOR_LR = 1e-4 # Discriminator's initial learning rate
__C.TRAIN.SNAPSHOT_INTERVAL = 5 # Every how many epochs to save checkpoints
__C.TRAIN.SNAPSHOT_RANGE = 10 # Keep how many checkpoints in total during training
'''
    GAN loss type
        This is not used in either pix2pixHD or BicycleGAN because 
          they have their own GAN criterion
'''
__C.TRAIN.LOSS_TYPE = 'HINGE' # GAN Loss type if GAN criterion not specified

__C.TRAIN.LR_DECAY_VALUE = 0.999 # Linearly decay to (1 - 0.999) * initial learning rate
__C.TRAIN.LR_DECAY_COUNT = 1.3e5 # From which step to start learning rate decay
'''
    Maximum training iteration steps
        If TOTAL_SIZE is specified none-zero, then MAX_COUNT is calculated 
            using it.
'''
__C.TRAIN.MAX_COUNT = 2.6e5

# Training weights attached to different losses
__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.NE = 1.0 # The weight attached to the regularizer loss
__C.TRAIN.COEFF.L1 = 1.0 # The weight attached to L1 regression loss
__C.TRAIN.COEFF.GAN = 1.0 # The weight attached to GAN loss

#--------------------------------------------------
# Regularizer hyperparameter setting
#--------------------------------------------------
__C.NOISE_EXAG = False # Use the regularizer or not
__C.NE_MARGIN = False # Whether to clip the regularizer loss using a margin
__C.NE_MARGIN_VALUE = 10.0 # What is the clip margin value
__C.FEAT_DIFF = False # Calculate the output difference using featrures from
                        # discriminators

#--------------------------------------------------
# Evaluation setting
#--------------------------------------------------
"""
    Infer(sample) how many images from a condition
    When training, it will also infer this many images to monitor training 
        process when every recording happens.
"""
__C.SAMPLE_NUM = 3

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v) and not (type(v).__name__ == "unicode"):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif old_type is int and type(a[k]) is float and \
                    k.find('COUNT') != -1:
                b[k] = int(v)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            if type(v).__name__ == "unicode":
                b[k] = str(v)
            else:
                b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
