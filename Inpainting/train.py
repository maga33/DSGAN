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
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
parser.add_argument('--dataset_name', type=str, default='img_align_celeba', help='name of the dataset')
parser.add_argument('--crop_mode', type=str, default='trans_ctr', help='[random | trans_ctr]')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=256, help='size of each image dimension')
parser.add_argument('--load_size', type=int, default=280, help='size of each image loading dimension')
parser.add_argument('--mask_size', type=int, default=128, help='size of random mask')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1500, help='interval between image sampling')
parser.add_argument('--snapshot_interval', type=int, default=1, help='interval between image sampling')
parser.add_argument('--vis_interval', type=int, default=500, help='interval between image sampling')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='location to checkpoint')
parser.add_argument('--noise_w', type=float, default=5, help='weights for diversity-encouraging term')
parser.add_argument('--feat_w', type=float, default=10, help='weights for diversity-encouraging term')
parser.add_argument('--noise_dim', type=int, default=32, help='dimmension for noise vector')
parser.add_argument('--no_noise', action='store_true', help='do not use the noise if specified')
parser.add_argument('--dist_measure', type=str, default='perceptual', help='[rgb | perceptual]')
parser.add_argument('--n_layers_G', type=int, default=3, help='number of layers in generator')
parser.add_argument('--n_layers_D', type=int, default=3, help='number of layers in discriminator')
parser.add_argument('--num_D', type=int, default=2, help='number of discriminators for multiscale PatchGAN')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# visualization
exp_name = 'CENoise_noiseDim_%d_lambda_%.3f_outputDist%s' % (opt.noise_dim, opt.noise_w, opt.dist_measure)
if not opt.crop_mode == 'random':
    exp_name += '_' + opt.crop_mode
checkpoint_dir = os.path.join(opt.checkpoint_dir, exp_name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

writer = SummaryWriter(log_dir=checkpoint_dir)

# Loss function
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
from models.losses import GANLoss
criterionGAN = GANLoss(use_lsgan=True, tensor=Tensor)
criterionFeat = torch.nn.L1Loss()

# Initialize generator and discriminator
from models.Generator_NET import GlobalTwoStreamGenerator
generator = GlobalTwoStreamGenerator(input_nc=3, n_downsampling=opt.n_layers_G, z_dim=opt.noise_dim)
from models.Discriminator_NET import MultiscaleDiscriminator
discriminator = MultiscaleDiscriminator(input_nc=3, n_layers=opt.n_layers_D, num_D=opt.num_D)

if cuda:
    torch.cuda.set_device(opt.gpu_id)
    generator.cuda()
    discriminator.cuda()
    criterionGAN.cuda()
    criterionFeat.cuda()

# Initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)

# Dataset loader
batch_size_test = 4
transforms_ = [ transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset("./data/%s" % opt.dataset_name, \
                                     img_size=opt.img_size, load_size=opt.load_size, mask_size=opt.mask_size, crop_mode=opt.crop_mode),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
test_dataloader = DataLoader(ImageDataset("./data/%s" % opt.dataset_name, \
                                          img_size=opt.img_size, load_size=opt.load_size, mask_size=opt.mask_size, mode='val'),
                        batch_size=batch_size_test, shuffle=True, num_workers=1)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def sample_test(test_num_noise=8):
    """
    Run inapinting
    Inputs:
        test_num_noise: number of latent codes to visualize
    Outputs:
        gen_mask: generated images
    """
    samples, masked_samples, i, masks = next(iter(test_dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    masks = Variable(masks.type(Tensor))

    # pad the gt for visualization
    i = i.repeat(test_num_noise)
    paded_samples = masked_samples.repeat(test_num_noise, 1, 1, 1)
    padded_masks = masks.repeat(test_num_noise, 1, 1, 1)
    noise = torch.randn(samples.size(0)*test_num_noise, opt.noise_dim, 1, 1)
    noise = Variable(noise.cuda())
    # Generate inpainted image
    generator.eval()
    gen_mask = generator(paded_samples, noise, padded_masks)
    generator.train()

    return gen_mask

def complete_img(masked_img, gen_patch, coords):
    """
    Construct inpainted images using the input and the predicted patch
    Inputs:
        masked_img: images with missing region
        gen_patch: predicted patch of the missing region
        coords: coordinate of missing region (xmin,ymin,xmax,ymax)
    Outputs:
        filled_img: completed image
    """
    i = coords[0]
    filled_img = masked_img.clone()
    filled_img[:, :, i:i+opt.mask_size, i:i+opt.mask_size] = gen_patch[:,:, i:i+opt.mask_size, i:i+opt.mask_size]
    return filled_img

def discriminate(netD, image, mask):
    """
    Wrapper function for discriminator
    Inputs:
        netD: discriminator
        image: input image
        mask: binary mask indicating the missing region
    Outputs:
        outputs from the discriminators
    """
    netD_in = image * mask.repeat(1, image.size(1), 1, 1)
    return netD(netD_in)

# -----------------
#  Run Training
# -----------------
total_iter = 0
for epoch in range(opt.n_epochs):
    for cnt, (imgs, masked_imgs, masked_parts, masks) in enumerate(dataloader):

        # Configure input
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        masked_parts = Variable(masked_parts.type(Tensor))
        masks = Variable(masks.type(Tensor))

        # double the batches
        imgs = torch.cat((imgs, imgs),dim=0)
        masked_imgs = torch.cat((masked_imgs, masked_imgs), dim=0)
        masked_parts = torch.cat((masked_parts, masked_parts), dim=0)
        masks = torch.cat((masks,masks), dim=0)
        # sample noises
        B = int(imgs.size(0)/2)
        noise = torch.randn(imgs.size(0), opt.noise_dim, 1, 1)
        noise = Variable(noise.cuda())

        # run generation
        gen_parts = generator(masked_imgs, noise, masks)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        pred_real = discriminate(discriminator, masked_parts, masks)
        pred_fake = discriminate(discriminator, gen_parts.detach(), masks)
        real_loss = criterionGAN(pred_real, True)
        fake_loss = criterionGAN(pred_fake, False)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Adversarial and pixelwise loss
        pred_fake = discriminate(discriminator, gen_parts, masks)
        g_adv = criterionGAN(pred_fake, True)
        g_feat = 0
        feat_weights = 4.0 / (opt.n_layers_D + 1)
        D_weights = 1.0 / opt.num_D
        for i in range(opt.num_D):
            for j in range(len(pred_fake[i])-1):
                g_feat += D_weights * feat_weights * \
                    criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * opt.feat_w

        # noise sensitivity loss
        if opt.dist_measure == 'rgb':
            g_noise_out_dist = torch.mean(torch.abs(gen_parts[:B] - gen_parts[B:]))
        elif opt.dist_measure == 'perceptual':
            g_noise_out_dist = 0
            for i in range(opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    g_noise_out_dist += D_weights * feat_weights * \
                        torch.mean(torch.abs(pred_fake[i][j][:B] - pred_fake[i][j][B:]).view(B,-1),dim=1)

        g_noise_z_dist = torch.mean(torch.abs(noise[:B] - noise[B:]).view(B,-1),dim=1)
        g_noise = torch.mean( g_noise_out_dist / g_noise_z_dist ) * opt.noise_w

        # Total loss
        g_loss = g_adv + g_feat - g_noise
        g_loss.backward()
        optimizer_G.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]" % (epoch, opt.n_epochs, cnt, len(dataloader),
                                                            d_loss.data[0], g_adv.data[0], g_feat.data[0]))

        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + cnt
        if batches_done % opt.vis_interval == 0:
            # record for visualization
            writer.add_scalar('d_loss', d_loss.data[0], batches_done)
            writer.add_scalar('g_loss', g_loss.data[0], batches_done)
            writer.add_scalar('g_adv', g_adv.data[0], batches_done)
            writer.add_scalar('g_noise', g_noise.data[0], batches_done)
            train_gt_img = make_grid(masked_imgs.cpu().data, normalize=True, scale_each=True, nrow=opt.batch_size)
            train_gen_parts = make_grid(gen_parts.cpu().data, normalize=True, scale_each=True, nrow=opt.batch_size)
            train_gt_parts = make_grid(masked_parts.cpu().data, normalize=True, scale_each=True, nrow=opt.batch_size)

            writer.add_image('gt_img', train_gt_img, batches_done)
            writer.add_image('pred_patch', train_gen_parts, batches_done)
            writer.add_image('gt_patch', train_gt_parts, batches_done)
            #test_img = save_sample(batches_done)

        if batches_done % opt.sample_interval == 0:
            test_gen_imgs = sample_test(8)
            test_gen_imgs = make_grid(test_gen_imgs.cpu().data, normalize=True, scale_each=True, nrow=batch_size_test)
            writer.add_image('pred_patch_test', test_gen_imgs, batches_done)
            # save the latest network
            save_path = os.path.join(checkpoint_dir, 'netG_latest.pth')
            torch.save(generator.state_dict(), save_path)

    if (epoch+1) % opt.snapshot_interval == 0:
        save_path = os.path.join(checkpoint_dir, 'netG_%d.pth' % epoch)
        torch.save(generator.state_dict(), save_path)
