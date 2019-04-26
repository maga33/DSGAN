import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets_mask import *
from models import *
from models.layer_util import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
parser.add_argument('--dataset_name', type=str, default='img_align_celeba', help='name of the dataset')
parser.add_argument('--crop_mode', type=str, default='trans_ctr', help='[random | trans_ctr]')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=256, help='size of each image dimension')
parser.add_argument('--load_size', type=int, default=280, help='size of each image loading dimension')
parser.add_argument('--mask_size', type=int, default=128, help='size of random mask')
parser.add_argument('--num_test', type=int, default=10, help='interval between image sampling')
parser.add_argument('--num_noise', type=int, default=4, help='interval between image sampling')
parser.add_argument('--noise_dim', type=int, default=32, help='dimmension for noise vector')
parser.add_argument('--latent_interval', type=int, default=8, help='dimmension for noise vector')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='location to checkpoint')
parser.add_argument('--epoch', type=str, default='latest', help='epoch of the checkpoint to load')
parser.add_argument('--exp_name', type=str, default='', help='name of the checkpoint')
parser.add_argument('--n_layers_G', type=int, default=3, help='number of layers in generator')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

batch_size=1
num_noise = opt.num_noise

# visualization
result_dir = os.path.join('results', opt.exp_name, 'latent_anal')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Initialize generator and discriminator
from models.Generator_NET import GlobalTwoStreamGenerator
generator = GlobalTwoStreamGenerator(input_nc=3, n_downsampling=opt.n_layers_G, z_dim=opt.noise_dim)

# load the network
checkpoint_dir = os.path.join('checkpoints', opt.exp_name)
save_path = os.path.join(checkpoint_dir, 'netG_%s.pth' % opt.epoch)
generator.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage))
generator.eval()


if cuda:
    torch.cuda.set_device(opt.gpu_id)
    generator.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Dataset loader
batch_size_test = 4
transforms_ = [ transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset("./data/%s" % opt.dataset_name, \
                                          img_size=opt.img_size, load_size=opt.load_size, mask_size=opt.mask_size, mode='val'),
                        batch_size=batch_size, shuffle=True, num_workers=1)



# fix the noise
noise = torch.zeros(batch_size*num_noise, opt.noise_dim, 1, 1)
noise = Variable(noise.cuda())

# run inference
# setup visualization
vis_pred = torch.zeros(num_noise+1, opt.num_test, 3, opt.img_size, opt.img_size)

for cnt, (imgs, masked_imgs, mask_coords, masks) in enumerate(dataloader):
    print('[%d/%d]' % (cnt, opt.num_test))
    if cnt >= opt.num_test:
        break

    # sample the latent code
    noise.data.normal_(0, 0.5)

    # Configure input
    imgs = Variable(imgs.type(Tensor))
    masked_imgs = Variable(masked_imgs.type(Tensor))
    masks = Variable(masks.type(Tensor))
    vis_pred[0,cnt,:,:,:] = masked_imgs.cpu().data[0]

    # increase the batches
    imgs = imgs.repeat(num_noise, 1, 1, 1)
    masked_imgs = masked_imgs.repeat(num_noise, 1, 1, 1)
    masks = masks.repeat(num_noise, 1, 1, 1)

    # forward through generator
    gen_parts = generator(masked_imgs, noise, masks)
    vis_pred[1:,cnt,:,:,:] = gen_parts.cpu().data

# stack for visualization
vis_pred = vis_pred.view(-1, 3, opt.img_size, opt.img_size)
vis_pred = save_image(vis_pred, os.path.join(result_dir, 'results.png'), \
                        normalize=True, scale_each=True, nrow=opt.num_test)

