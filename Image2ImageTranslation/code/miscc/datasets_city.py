from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from miscc.config import cfg

class Dataset(data.Dataset): 
       
    def __init__(self, data_dir, split='train', imsize=512, config=None, 
            pre_set_list=[]):
        if config is None:
            self.config = cfg
        else:
            self.config = config


        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir
        self.split = split
        self.split_dir = os.path.join(data_dir, split)

        if len(pre_set_list) == 0:
            with open(os.path.join(data_dir, "%s_filenames.pickle" % split), 
                    'rb') as f:
                self.filenames = pickle.load(f)
        else:
            self.filenames = pre_set_list



    def __getitem__(self, index):
        file_name = self.filenames[index]
        # captions = self.captions[key]
        img_path = os.path.join(self.split_dir, 
                '%s_leftImg8bit.png' % file_name)
        label_map_path = os.path.join(self.split_dir, 
                '%s_gtFine_labelIds.png' % file_name)
        inst_map_path = os.path.join(self.split_dir, 
                '%s_gtFine_instanceIds.png' % file_name
                )


        img = Image.open(img_path).convert('RGB')
        label_map = Image.open(label_map_path)
        inst_map = Image.open(inst_map_path)


        ############################################################
        # transoform RGB image, and to Tensor
        # note should use the same transoform param dictionary
        ############################################################
        trans_params = get_param_dict(self.config)

        img_transform = get_transforms(trans_params, method=Image.BILINEAR, 
                normalize=True, split=self.split)
        img_tensor = img_transform(img)

        if cfg.CLASS_NUM == 1:
            lm_transform = get_transforms(trans_params, method=Image.NEAREST, 
                    split=self.split)
            label_map_tensor = lm_transform(label_map)
        else:
            lm_transform = get_transforms(trans_params, method=Image.NEAREST, 
                    normalize=False, split=self.split)
            label_map_tensor = lm_transform(label_map) * 255.0
            
        inst_map_tensor = lm_transform(inst_map)

        mask_tensor = label_map_tensor

        return img_tensor, mask_tensor, inst_map_tensor, file_name


    def __len__(self):
        return len(self.filenames)

"""assitance function for transforming images"""
def construct_one_hot(label_map_tensor):
    lm_size = label_map_tensor.size()
    oneHot_size = (cfg.CLASS_NUM, lm_size[1], lm_size[2])
    mask_tensor = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
    mask_tensor = mask_tensor.scatter_(0, label_map_tensor.long(), 1.0)
    # the 0 represents unlabled. 
    return mask_tensor


def get_param_dict(config):
    params = {}
    params['flip'] = np.random.rand(1)[0] > 0.5


    diff_length, diff_width = config.LOAD_SIZE - config.IMSIZE, \
            config.IM_RATIO * config.LOAD_SIZE - config.IM_RATIO *\
            config.IMSIZE


    pos_x = np.random.randint(0, diff_width)
    pos_y = np.random.randint(0, diff_length)


    params['crop_pos'] = (pos_x, pos_y)
    params['load_size'] = config.LOAD_SIZE
    params['img_size'] = config.IMSIZE

    return params

"""if we train higher resolution image based on lower resolution pre
    -trained results, we should scale the dict"""
def scale_param_dict(params):
    scaled_params = {}
    pos = params['crop_pos']
    scaled_params['crop_pos'] = (pos[0] * 2, pos[1] * 2)
    scaled_params['load_size'] = params['load_size'] * 2
    scaled_params['img_size'] = params['img_size'] * 2
    scaled_params['flip'] = params['flip']

    return scaled_params

def get_transforms(params, method=Image.BILINEAR, normalize=False, split=None):


    if split == 'train':
        trans_list = [
                        transforms.Scale([params['load_size'], 
                            cfg.IM_RATIO * params['load_size']]
                             , method),
                        transforms.Lambda(lambda img: __flip(img, params['flip'])),
                        transforms.Lambda(lambda img: __crop(img, params['crop_pos'], 
                            params['img_size'])),
                        transforms.ToTensor(),
                    ]
    elif split == 'val':
        trans_list = \
                [
                    transforms.Scale([params['img_size'], 
                        cfg.IM_RATIO * params['img_size']], 
                        method),
                    transforms.ToTensor()
                ]


    if normalize:
        trans_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))


    trans = transforms.Compose(trans_list)


    return trans


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw , th = cfg.IM_RATIO * size, size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))


    return img


def classwise_noise(mask_var_cpu, n_num):
    # n_num, noise dim
    b_mask_numpy = mask_var_cpu.clone().numpy()
    bool_mask = b_mask_numpy.astype(np.bool)
    shape = b_mask_numpy.shape
    class_num = shape[0]


    noise_block = np.zeros((n_num, shape[1], shape[2]))


    for j in range(class_num):
        mask_index = bool_mask[j]
        if np.any(mask_index):
            noise_block[:, mask_index] = \
                    np.random.randn(n_num, 1)


    noise_block_tensor = torch.from_numpy(noise_block).clone()


    return noise_block_tensor.float()


def get_edges(t):
    edge = torch.ByteTensor(t.size()).zero_()
    edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    return edge.float()
