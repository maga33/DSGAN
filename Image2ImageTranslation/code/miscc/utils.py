import os
import errno
from copy import deepcopy
import argparse
import functools
import math

import numpy as np
import scipy
from scipy.io import loadmat
from scipy import ndimage
import PIL.Image as Image
import numpy as np


from miscc.config import cfg
from miscc.layer_utils import Identity

from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable

# Set the color map for displaying the segmentations, Using VOC color setting
voc_cmap = loadmat("../data/voc_gt_cmap.mat")
voc_mapper = (voc_cmap['cmap'][1:81]*255).astype(np.uint8)
voc_mapper[[0, 14]] = voc_mapper[[14, 0]]

#--------------------------------------------------
# Load Networks
#--------------------------------------------------
def load_networks(model_dir):
    """
        Load Networks:
        1. Select Generator/Discriminator pair with BicycleGAN or pix2pixHD 
            setting.
        2. If there is previous checkpoints in 'model_dir', unless you specify
            the specific checkpoint name in 'cfg.NET_G', load the latest 
            checkpoint.
        3. If not in training mode, only load the generator.
    """
    if cfg.DATASET_NAME.lower().find('city') != -1: input_ch = cfg.CLASS_NUM + 1;
    else: input_ch = cfg.CLASS_NUM;

    if cfg.G_TYPE == 'GEN_HD':
        # Use the Global Generator in  pix2pixHD as the generator
        from exterior_models.pix2pixHD.networks import GlobalGenerator
        norm_layer = nn.InstanceNorm2d

        netG = GlobalGenerator(input_ch, 3, ngf=32, 
                n_downsampling=4, n_blocks=9, norm_layer=norm_layer)

    elif cfg.G_TYPE == 'GEN_Bicycle':
        # Use The Generator in BicycleGAN as the generator
        from  exterior_models.bicycleGAN.networks import G_Unet_add_all

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        nl = functools.partial(nn.ReLU, inplace=True)

        netG = G_Unet_add_all(input_ch, 3, nz=cfg.Z_NUM, 
                num_downs=8, ngf=64, norm_layer=norm_layer, 
                nl_layer=nl, use_dropout=True, gpu_ids=[], upsample='basic')

    print("\033[1;31m Generator Structure: \033[0m")
    print(netG)

    if cfg.TRAIN.FLAG:
        if cfg.DIS_HD or cfg.D_TYPE == 'DIS_HD':
            from exterior_models.pix2pixHD.networks import MultiscaleDiscriminator

            netD = MultiscaleDiscriminator(cfg.CLASS_NUM + 3, ndf=64, 
                    n_layers=3, norm_layer=norm_layer, 
                    use_sigmoid=False, num_D=cfg.NUM_D, getIntermFeat=True)

        elif cfg.D_TYPE == 'DIS_Bicycle':
            from exterior_models.bicycleGAN.networks import D_NLayersMulti

            netD = D_NLayersMulti(cfg.CLASS_NUM + 3, ndf=64, n_layers=3, 
                    norm_layer=norm_layer, use_sigmoid=False, gpu_ids=[], 
                    num_D=2)

        print("\033[1;31m Discrminator Structure: \033[0m")
        print(netD)

    else: 
        netD = None

    netG_path = find_latest_model_file(model_dir, cfg, 'netG')

    if netG_path != '':
        load_model(netG, netG_path)
        print('Load Generator from: ', netG_path)
        if netD is not None:
            net_D_path = os.path.join(model_dir, 'netD_epoch_last.pth')
            load_model(netD, net_D_path)
            print('Load Discrminator from: ', net_D_path)

    if cfg.CUDA:
        netG.cuda()
        if netD is not None: netD.cuda();

    return netG, netD

def cat_two_tensor(tensor1, tensor2):
    # concatenate 2-dim tensor(tensor2) to 4-dim tensor(tensor1)
    t2_size = tensor2.size()
    t1_size = tensor1.size()
    result = torch.cat(
            (tensor1, tensor2.unsqueeze(dim=2).unsqueeze(dim=2).\
                    expand(t2_size[0], t2_size[1], t1_size[2],
                        t1_size[3])),
                    dim=1
            )
    return result

def multitensor2img(masks_tensor, imtype=np.uint8):
    """
        If the tensor channel is more than one, 
            aggregate it using VOC color map
    """
    masks_numpy = masks_tensor.cpu().float().numpy()
    nc = masks_numpy.shape[1]
    batchSize = masks_numpy.shape[0]

    pad_imgs_list = []
    if nc != 1:
        for j in range(batchSize):
            new_pad = np.ones([masks_numpy.shape[2], masks_numpy.shape[3], 3], dtype='float')
            for i in range(nc):
                # the hard threshold for generating the mask from 
                # SOFTMAX SCORE is 0.5
                mask = masks_numpy[j][i] == 1
                new_pad[mask,:] = voc_mapper[i].astype(np.float32) / 255

            new_pad = new_pad * 255
            new_pad = new_pad.astype(imtype)
            pad_imgs_list.append(new_pad)
        

        batched_pad_imgs = np.stack(pad_imgs_list, axis=0)
        batched_pad_imgs = np.transpose(batched_pad_imgs, (0, 3, 1, 2))
        result = torch.ByteTensor(batched_pad_imgs).clone()
    else:
        result = (masks_tensor * 255).byte()

    return result

def save_img_results(imgs_dict, r_tocken, image_dir, nrow=1):
    """
        Arrage every image tensors in 'imgs_dict' into a grid

        Input:

            imgs_dict: Ordered dictionary, batched image tensors and names in it
            r_tocken: Recording tocken
            image_dir: The directory to save the images for web browsing
            nrow: How many rows in the grid, a parameter in 
                torchvision.utils.make_grid

        Output:
            
            output_grids: Grids of images
            img_names: The names of image grids

    """
    mean = torch.FloatTensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.FloatTensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

    output_grids = []
    img_names = []
    for key, image_data in imgs_dict.items():
        if isinstance(image_data, Variable):
            image_data = image_data.data.cpu()
        else:
            image_data = image_data.cpu()

        if image_data.size(1) == 3:
            image_data = image_data * std + mean
        elif image_data.size(1) > 3:
            image_data = multitensor2img(image_data)
        elif cfg.CLASS_NUM == 1 and cfg.DATASET_NAME == 'cityscapes':
                image_data = image_data * std + mean

        output_grids.append(vutils.make_grid(image_data, 
            nrow=nrow, normalize=False))
        im_array = output_grids[-1].mul(255).clamp(0, 255).byte().permute(1,
                2, 0).numpy()
        im_to_save = Image.fromarray(im_array)
        if isinstance(r_tocken, (int, float, long)):
            img_names.append('%s_epoch_%03d.png' % (key, r_tocken))
        else:
            img_names.append('%s_%s.png' % (key, r_tocken))

        im_to_save.save(os.path.join(image_dir, img_names[-1]))
    
    return output_grids, img_names


def save_model(netG, netD, netE, epoch, interval, model_dir):
    """ Save Models """
    torch.save(
        netG.cpu().state_dict(),
        '%s/netG_latest_%d_inter_%d.pth' % (model_dir, epoch, interval))
    if cfg.CUDA:
        netG.cuda()

    if netD is not None:
        torch.save(
            netD.cpu().state_dict(),
            '%s/netD_epoch_last.pth' % (model_dir))
        if cfg.CUDA:
            netD.cuda()

    if netE is not None:
        torch.save(
            netE.cpu().state_dict(),
            '%s/netE_epoch_last.pth' % (model_dir))
        if cfg.CUDA:
            netE.cuda()

    print('Saved models')

def load_model(network, save_path):
    """ Load and match parameters, give warnings if not matched"""
    if not os.path.isfile(save_path):
        print('%s not exists yet!' % save_path)
    else:
        #network.load_state_dict(torch.load(save_path))
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            pretrained_dict = torch.load(save_path)
            model_dict = network.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                network.load_state_dict(pretrained_dict)
                print('Pretrained network has excessive layers; Only loading layers that are used')
            except:
                print('Pretrained network has fewer layers; The following are not initialized')
                from sets import Set
                not_initialized = Set()
                for k, v in pretrained_dict.items():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.add(k.split('.')[0])

                print(sorted(not_initialized))
                network.load_state_dict(model_dict)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def find_latest_model_file(path, cfg, keyword=None):
    """Find the lastest create VAE checkpoint path if any,
    otherwise return ''.
    Also allows to specify certain checkpoint manually by cfg.NET_VAE
    """
    if cfg.NET_G != '' and keyword == 'netG':
        return os.path.join(path, cfg.NET_G)

    try: 
        files = os.listdir(path)
    except OSError: return '';

    if len(files) == 0: return '';

    paths = [os.path.join(path, basename) for basename 
            in files if basename.find('netG') !=-1]
    netG_path = max(paths, key=os.path.getctime)

    return  netG_path

def switch_grad(net, switch_flag):
    """
        for '.u' we always want the requires_grad to be False, because 
        the update is not got by gradients descent.
        Instead it is got by power iteration.
    """
    for key, var in net.state_dict(keep_vars=True).items():
        if not key.endswith(".u"):
            var.requires_grad = switch_flag

def compute_digits_difference(digits1, digits2, weight=1.0):
    feat_diff = 0
    feat_weights = 4.0 / (3 + 1) # 3 layers's discrminator
    D_weights = 1.0 / float(cfg.NUM_D) # number of discrminator
    for i in range(cfg.NUM_D):
        for j in range(len(digits2[i])-1):
            feat_diff += D_weights * feat_weights * \
                F.l1_loss(digits2[i][j], 
                        digits1[i][j].detach()) * weight
    return feat_diff

def compute_gan_loss(real_digits, fake_digits, loss_type=None, 
        loss_at='None', gan_cri=None):
    """
        Compute GAN Loss
        
        If 'gan_cri': GAN criterion is not None, use that, 
            else, refer to the 'loss_type'

        Output:

        errG: Generator GAN loss
        errD: Discrminator GAN loss
        errG_feat: Feature matching loss in pix2pixHD 
    """
    errD = None
    errG = None
    errG_feat = None
    if gan_cri is not None:
        if loss_at == 'D':
            errD = (gan_cri(real_digits, True) \
                    + gan_cri(fake_digits, False)) * 0.5
        elif loss_at == 'G':
            errG = gan_cri(fake_digits, True)
            if cfg.D_TYPE == 'DIS_HD':
                errG_feat = compute_digits_differnce(real_digits, fake_digits, 
                        weight=10.0)

    elif loss_type == 'HINGE':
        if loss_at == 'D':
            errD =  torch.clamp(1 - real_digits, min=0) + \
                    torch.clamp(1 + fake_digits, min=0)
            
            errD = errD.mean()
        elif loss_at == 'G':
            errG = - fake_digits.mean()
    elif loss_type == 'KL':
        if loss_at == 'D':
            errD = F.logsigmoid(real_digits) +\
                    F.logsigmoid(- fake_digits)
            errD = - errD.mean()
        else:
            errG = - F.logsigmoid(fake_digits).mean()

    return errG, errD, errG_feat

def var_and_cuda(tensor_list):
    var_list = []
    for i, x in enumerate(tensor_list):
        if i == 1 and cfg.CLASS_NUM > 3:
            size = x.size()
            oneHot_size = (size[0], cfg.CLASS_NUM, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, x.data.long().cuda(), 1.0)
            var_list.append(input_label)
            continue
            
        if x is None:
            var_list.append(None)
            continue

        if cfg.CUDA:
            x = x.cuda()

        var_list.append(x)

    return var_list

# Make all layers to be spectral normalization layer
def add_sn(m):

    for name, c in m.named_children():
        m.add_module(name, add_sn(c))    

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m

def cfg_args_exchange(cfg, args):
    if args.no_cuda:
        cfg.CUDA = False
    if args.snapshot_range != -1:
        cfg.TRAIN.SNAPSHOT_RANGE = args.snapshot_range
    if args.snapshot_interval != -1:
        cfg.TRAIN.SNAPSHOT_INTERVAL = args.snapshot_interval
    if args.gpu_id != '0':
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if cfg.D_TYPE == 'DIS_HD':
        cfg.DIS_HD = True
    if args.margin:
        cfg.NE_MARGIN = True
    if args.margin_value != -1:
        cfg.NE_MARGIN_VALUE = args.margin_value
    if args.ne_weight != -1:
        cfg.TRAIN.COEFF.NE = args.ne_weight
    if args.gan_weight != -1:
        cfg.TRAIN.COEFF.GAN = args.gan_weight
    if args.max_epoch != -1:
        cfg.TRAIN.MAX_EPOCH = args.max_epoch
    if args.gen_paths == "output":
        args.gen_paths = ""
    if args.feat_diff:
        cfg.FEAT_DIFF = True
    if args.z_num != -1:
        cfg.Z_NUM = args.z_num
    if cfg.TRAIN.COEFF.NE == 0.0 or args.no_ne:
        cfg.NOISE_EXAG = False
    if args.batch_size != 1:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.no_l1_loss:
        cfg.L1_LOSS = False
    if args.gen_lr != -1.0:
        cfg.TRAIN.GENERATOR_LR = args.gen_lr
    if args.dis_lr != -1.0:
        cfg.TRAIN.DISCRIMINATOR_LR = args.dis_lr
    
    """
        If TOTAL_SIZE (the number of dataset samples) is specified
            calculate the COUNT using this
    """
    if cfg.TOTAL_SIZE > 0:
        cfg.TRAIN.MAX_COUNT = int(math.floor(cfg.TOTAL_SIZE / float(
            cfg.TRAIN.BATCH_SIZE))) * cfg.TRAIN.MAX_EPOCH
        cfg.TRAIN.LR_DECAY_COUNT = int(cfg.TRAIN.MAX_COUNT / 2.0)

    if args.net_D != '':
        cfg.NET_D = args.net_D


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file', default='birds_stage1.yml', 
            type=str,
            help="optional config file")
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0', 
            help="GPU id number")
    parser.add_argument('--no_cuda', action="store_true", 
            help="Not use GPU, usually use this to overwrite cfg.CUDA flag during training")

    parser.add_argument('--data_dir', dest='data_dir', type=str, default='', 
            help="Dataset folder")
    parser.add_argument('--debug_flag', action='store_true', 
            help="To run the model in debug mode or not")
    parser.add_argument('--output_dir', type=str, default='')
    
    parser.add_argument('--margin', action='store_true', 
            help="Use margin to clip the regularizer loss or not")
    parser.add_argument('--margin_value', type=float, default=-1, 
            help="Margin clip value")
    parser.add_argument('--ne_weight', type=float, default=-1, 
            help="The weight attached to regularizer loss")
    parser.add_argument('--gan_weight', type=float, default=-1, 
            help="The weight attached to the GAN loss")
    parser.add_argument('--max_epoch', type=int, default=-1, 
            help="Maximum training epoch number")
    parser.add_argument('--gen_paths', type=str, default="", 
            help="The generated images paths, use this to do FID evaluation")
    parser.add_argument('--feat_diff', action='store_true', 
            help="Use the difference among the discriminator features")
    parser.add_argument('--z_num', type=int, default=-1)
    parser.add_argument('--no_l1_loss', action='store_true')
    parser.add_argument('--no_ne', action='store_true')
    parser.add_argument('--ne_grad', action='store_true')
    parser.add_argument('--total_size', type=int, default=0)
    parser.add_argument('--gen_lr', type=float, default=-1.0, 
            help="Generator learning rate")
    parser.add_argument('--dis_lr', type=float, default=-1.0, 
            help="Discriminator learning rate")

    #--------------------------------------------------
    # Evaluation flags
    #--------------------------------------------------
    # general eval flags
    parser.add_argument('--eval', action='store_true', 
            help="General evaluation flag.")
    parser.add_argument('--eval_all', action='store_true', 
            help="Evaluate all saved checkpoints.")
    parser.add_argument('--epoch_to_eval', type=int, default=5, 
            help="The maximum number of epochs we want to evaluate, "\
                    "Use together with 'eval_all' flag.")

    parser.add_argument('--inter_eval', action='store_true', 
            help="Get interpolated evaluation results")
    parser.add_argument('--LPIPS_eval', action='store_true', 
            help="Get LPIPS evaluation results")
    parser.add_argument('--FID_eval', action='store_true', 
            help="General FID evaluation flag")
    parser.add_argument('--FID_eval_20', action='store_true', 
            help="Do FID evaluation 20 times")

    parser.add_argument('--sample_num', type=int, default=20, 
            help="For each input condition, how many to infer")
    parser.add_argument('--eval_num', type=int, default=100)
    parser.add_argument('--try_num', type=int, default=20)


    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--net_G', type=str, default='', 
            help="Generator checkpoint file name")
    parser.add_argument('--net_D', type=str, default='', 
            help="Discriminator checkpoint file name")
    parser.add_argument('--snapshot_range', type=int, default=-1, 
            help="How many epochs we want to keep in the hard disk.")
    parser.add_argument('--snapshot_interval', type=int, default=-1, 
            help="Every how many epochs we want to save checkpoints.")

    parser.add_argument('--max_count', type=float, default=-1, 
            help="Maximum iteration number to train the model")
    parser.add_argument('--lr_decay_count', type=float, default=-1)

    parser.add_argument('--manualSeed', type=int, default=0,
            help="manual seed")

    args = parser.parse_args()
    return args


