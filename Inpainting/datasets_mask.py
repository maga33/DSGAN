import glob
import random
import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, img_size=128, load_size=None, mask_size=64, mode='train', crop_mode='random'):
        self.img_size = img_size
        self.load_size = load_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob('%s/*.jpg' % root))
        self.files = self.files[:-4000] if mode == 'train' else self.files[-4000:]
        self.crop_mode=crop_mode
        if crop_mode=='random' or crop_mode=='none' or mode=='val':
            transforms_ = [ transforms.Resize((img_size, img_size), Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
        else:
            transforms_ = [ transforms.Resize((load_size, load_size), Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
        self.transform = transforms.Compose(transforms_)


    def apply_random_mask(self, img):
        """Randomly masks image"""
        # select the corner
        x1 = np.random.randint(0, np.maximum(0, self.load_size-self.img_size))
        y1 = np.random.randint(0, np.maximum(0, self.load_size-self.img_size))
        cropped_img = img[:, y1:y1+self.img_size, x1:x1+self.img_size].clone()
        #
        y1, x1 = np.random.randint(0, self.img_size-self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        mask = torch.zeros(1, self.img_size, self.img_size)
        mask[0, y1:y2, x1:x2] = 1
        masked_part = cropped_img[:, y1:y2, x1:x2]
        masked_img = cropped_img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part, mask

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i:i+self.mask_size, i:i+self.mask_size] = 1
        mask = torch.zeros(1, self.img_size, self.img_size)
        mask[0, i:i+self.mask_size, i:i+self.mask_size] = 1

        return masked_img, i, mask

    def apply_translation_ctr_mask(self, img):
        # select the corner
        x1 = np.random.randint(0, np.maximum(0, self.load_size-self.img_size))
        y1 = np.random.randint(0, np.maximum(0, self.load_size-self.img_size))
        cropped_img = img[:, y1:y1+self.img_size, x1:x1+self.img_size].clone()
        # gen masked image and parts
        ctr = (self.load_size - self.mask_size) // 2
        ctr_x, ctr_y = (ctr-x1, ctr-y1)
        masked_img = cropped_img.clone()
        masked_img[:, ctr_y:ctr_y+self.mask_size, ctr_x:ctr_x+self.mask_size] = 1
        mask = torch.zeros(1, self.img_size, self.img_size)
        mask[0, ctr_y:ctr_y+self.mask_size, ctr_x:ctr_x+self.mask_size] = 1
        return cropped_img, masked_img, cropped_img, mask

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == 'train':
            if self.crop_mode=='random':
                # For training data perform random mask
                img, masked_img, aux, mask = self.apply_random_mask(img)
            elif self.crop_mode=='trans_ctr':
                img, masked_img, aux, mask = self.apply_translation_ctr_mask(img)
            elif self.crop_mode == 'none':
                masked_img, aux, mask = self.apply_center_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux, mask = self.apply_center_mask(img)

        return img, masked_img, aux, mask

    def __len__(self):
        return len(self.files)
