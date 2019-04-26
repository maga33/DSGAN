from __future__ import print_function
from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image

import torchfile
from tqdm import tqdm
import json
import os
import time
import datetime
## for writing videos
import cv2
from collections import OrderedDict

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import save_img_results, save_model, load_model
from miscc.utils import load_networks
from miscc.utils import compute_gan_loss, compute_digits_difference
from miscc.utils import var_and_cuda
import miscc.html as html
from miscc.datasets_city import get_edges

class GANTrainer(object):
    def __init__(self, output_dir, cfg_path=None):
        self.__set_directory(output_dir)
        self.cfg_path = cfg_path

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.snapshot_range = cfg.TRAIN.SNAPSHOT_RANGE
        
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.img_sz = cfg.IMSIZE
        if cfg.CUDA:
            torch.cuda.set_device(self.gpus[0])
            self.tensor = torch.cuda.FloatTensor
        else:
            self.tensor = torch.FloatTensor

        cudnn.benchmark = True

    def train(self, data_loader):
        netG, netD = load_networks(self.model_dir)

        # Different GAN criterion according to the baseline model setting
        criterionGAN = None
        if cfg.DIS_HD:
            from exterior_models.pix2pixHD.networks import GANLoss
            criterionGAN = GANLoss(use_lsgan=True, tensor=self.tensor)
        elif cfg.D_TYPE.find('DIS_Bicycle') != -1:
            from exterior_models.bicycleGAN.networks import GANLoss
            criterionGAN = GANLoss(mse_loss=True, tensor=self.tensor)
        else:
            raise NotImplementedError, "Unknown GAN loss"

        # If in debug mode, only try to run 20 iterations
        if len(data_loader) > 20: debug_count = 20;
        else: debug_count = len(data_loader) / 2;

        #--------------------------------------------------
        # Initialize recorder, html and optimizers
        #--------------------------------------------------
        # tensorflow and html records initialization
        from tensorboardX import SummaryWriter
        # Experiment log using tensorboard
        self.summary_writer = SummaryWriter(self.log_dir)
        # Experiment log in the form of a simple webpage
        webpage = html.HTML(self.output_dir, 'Experiment name = train')
        self.epoch_compare = -1 

        lr_decay_step = cfg.TRAIN.LR_DECAY_COUNT
        max_count = cfg.TRAIN.MAX_COUNT
        loss_type = cfg.TRAIN.LOSS_TYPE

        init_gen_lr = cfg.TRAIN.GENERATOR_LR
        init_dis_lr = cfg.TRAIN.DISCRIMINATOR_LR

        # optimizer initialization
        optimizerG, optimizerD = self.get_optimizer(
                netG, netD, init_gen_lr, init_dis_lr)

        #--------------------------------------------------
        # To read preivous counting variable if any:
        #--------------------------------------------------
        count_var_path = os.path.join(self.output_dir, "count_var.json")
        if os.path.isfile(count_var_path):
            with open(count_var_path, 'r') as f:
                count_dict = json.load(f)
            count = count_dict['count']
            start_epoch = count_dict['start_epoch']
        else:
            count = 0
            start_epoch = 0

        #--------------------------------------------------
        # Additional Loss apart from GAN loss
        #--------------------------------------------------
        # Possible L1 Loss
        if cfg.L1_LOSS: l1_criterion = nn.L1Loss();
        # Possible VGG Loss
        if cfg.VGG_LOSS:
            from exterior_models.pix2pixHD.networks import VGG_LOSS
            self.criterionVGG = VGGLoss()

        lr_decay_value = cfg.TRAIN.LR_DECAY_VALUE

        #--------------------------------------------------
        # Main training loop
        #--------------------------------------------------
        for epoch in range(start_epoch, self.max_epoch):

            if count > max_count: break;
            start_t = time.time()

            #--------------------------------------------------
            # (0) Updating the parameters
            #--------------------------------------------------
            for i, data in enumerate(data_loader, 0):

                self.__update_lr([optimizerG, optimizerD], 
                        [init_gen_lr, init_dis_lr], count, epoch, 
                        lr_decay_step=lr_decay_step, 
                        lr_decay_value=lr_decay_value,
                        max_epoch=self.max_epoch, max_count=max_count,
                        gamma=0.97, decay_type='linear')
                #--------------------------------------------------
                # (1) Prepare training data
                #--------------------------------------------------
                
                data = data[:-1] # elieminate the final term: file name
                # If using edge map 
                if data[2].dim() > 2: data[2] = get_edges(data[2]);

                real_imgs, mask_tensor, edge_tensor = var_and_cuda(data)
                dis_mask_tensor = mask_tensor

                if edge_tensor.dim() == 2: edge_tensor = None;

                #--------------------------------------------------
                # (2) Generate fake images
                #--------------------------------------------------
                inputs = (mask_tensor, edge_tensor)
                gen_imgs, z_var = self._parallel_wrapper(netG, inputs)

                #--------------------------------------------------
                # (3) Update D network
                #--------------------------------------------------
                netD.zero_grad()
                fake_digits = netD(gen_imgs.detach(), dis_mask_tensor)
                real_digits = netD(real_imgs, dis_mask_tensor)
                _, errD, _ = compute_gan_loss(real_digits, fake_digits, 
                        loss_type=loss_type, loss_at='D', 
                        gan_cri=criterionGAN)

                errD.backward()
                optimizerD.step()

                #--------------------------------------------------
                # (4) Update G network
                #--------------------------------------------------
                fake_digits = netD(gen_imgs, dis_mask_tensor)
                errG, _, _ = compute_gan_loss(real_digits, fake_digits, 
                        loss_type=loss_type, loss_at = 'G',
                        gan_cri=criterionGAN)

                #--------------------------------------------------
                # (5) Additional feedforward for regularization
                #--------------------------------------------------
                if cfg.NOISE_EXAG: 
                    errNE, gen_imgs_eg, z_var_eg \
                            = self.errE_calculate(netG, netD, inputs, gen_imgs,
                            fake_digits, z_var, dis_mask_tensor)

                # Ppossible L1 Loss
                if cfg.L1_LOSS:
                    errL1 = l1_criterion(gen_imgs, real_imgs)
                else:
                    errL1 = torch.FloatTensor([0,])
                    if cfg.CUDA: errL1 = errL1.cuda();
                # Possible VGG Loss
                if cfg.VGG_LOSS:
                    errG_vgg = self.criterionVGG(gen_imgs, real_imgs) * 10.0
                else:
                    errG_vgg = torch.FloatTensor([0,])
                    if cfg.CUDA: errG_vgg = errG_vgg.cuda()

                errG_total = errG + cfg.TRAIN.COEFF.NE * errNE +\
                        cfg.TRAIN.COEFF.L1 * errL1 + errG_vgg

                errG_total.backward()
                optimizerG.step()
                netG.zero_grad()

                #--------------------------------------------------
                # (6) Record information
                #--------------------------------------------------
                # Update the step
                count = count + 1

                # Every 100 steps or end of debugging process record the results
                if count % 100 == 0 and count != 0 or (cfg.DEBUG_FLAG and count \
                        == debug_count):
                    netG.eval()

                    # Dump the iteration count
                    count_dict = {}
                    count_dict['count'] = count
                    count_dict['start_epoch'] = epoch
                    
                    with open(count_var_path, 'w') as f:
                        json.dump(count_dict, f, indent=4)

                    # GAN Loss
                    error_collection = OrderedDict([
                        ('D_loss', errD.item()),
                        ('G_loss', errG.item()),
                        ])
                    
                    self.summary_writer.add_scalars(
                            'data/gan_losses', error_collection, count)

                    if cfg.NOISE_EXAG: 
                        self.summary_writer.add_scalar(
                            'data/NE_loss', errNE.item(), count)

                    if cfg.L1_LOSS:
                        self.summary_writer.add_scalar(
                            'data/l1_loss', errL1.item(), count)

                    if cfg.VGG_LOSS:
                        self.summary_writer.add_scalar(
                            'data/vgg_loss', errG_vgg.item(), count)

                    # Save the image results for each epoch
                    # Infer several for a single condition input
                    self.infer_and_save_imgs(netG, real_imgs, mask_tensor, 
                            edge_tensor, epoch, webpage=webpage)

                    if cfg.DEBUG_FLAG:
                        print("\033[1;31m Debug finished. \033[0m")
                        break

                    netG.train()

            end_t = time.time()
            print('''[%d/%d][%d/%d] Loss_D: %.4f, Loss_G: %.4f, 
            Loss_NE: %.4f, L1_Loss: %.4f, Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch, i, len(data_loader),
                     errD.item(), errG.item(), errNE.item(), errL1.item(),
                     end_t - start_t))

            #--------------------------------------------------
            # (7) Save the model
            #--------------------------------------------------
            if epoch % int(self.snapshot_interval) == 0:
                latest_count = (epoch / int(self.snapshot_interval)) % \
                        self.snapshot_range
                save_model(netG, netD, None, latest_count, self.snapshot_interval,
                        self.model_dir)

            if cfg.DEBUG_FLAG:
                print("debugging exists")
                exit()

        self.summary_writer.close()
        webpage.save()

    def __update_lr(self, optimizer_list, initial_lr_list, count, epoch,
            lr_decay_step=None, max_epoch=None, decay_type=None, 
            lr_decay_value=None, max_count=None, gamma=None):
        """Apply linear learning rate decay"""

        for optimizer_now, init_lr in zip(optimizer_list, initial_lr_list):

            # If applied learning rate decay
            if cfg.NO_DECAY:
                break

            lr_now = init_lr * (1 - float(count - int(max_count / 2.0))\
                    / (int(max_count / 2.0) + 1))

            for param_group in optimizer_now.param_groups:
                param_group['lr'] = lr_now


    def prepare_eval(self, eval_name=None):
        eval_dir = os.path.join(self.output_dir, eval_name)
        eval_img_dir = os.path.join(eval_dir, 'Image')
        mkdir_p(eval_dir)
        mkdir_p(eval_img_dir)

        webpage = html.HTML(eval_dir, 'Experiment name = %s' % eval_name)
        self.image_dir = eval_img_dir # change the image saving dir
        self.epoch_compare = ''
        
        return webpage

    def sample(self, data_loader, eval_num=100, eval_name='eval', 
            pre_Z_collect = None, ex_stamp=''):
        """
            Sample images from trained generators

            Parameters:
            eval_num: Total number of samples needed to be evaluated
            eval_name: Evaluation name for different kinds of evaluation
            pre_Z_collect: Previous latent vector collection. We can use this 
               to continue to interpolate some latent vectors or sample on certian 
               latent vectors.
            ex_stamp: Extra stamp on saved generated image files
        """
        webpage = self.prepare_eval(eval_name=eval_name)

        netG, _ = load_networks(self.model_dir)
        netG.eval()
        pbar = tqdm(total=eval_num)

        print("\033[1;31m Doing Evaluation, Name: **%s**. \033[0m" % eval_name)
        # one_data = None
        for i, data in enumerate(data_loader):
            pbar.update(cfg.TRAIN.BATCH_SIZE)

            if eval_name.find('inter') != -1:
                # Spherical interpolation
                if pre_Z_collect is None:
                    pre_Z_collect = Variable(torch.randn(2, cfg.Z_NUM))
                # If not none we want it to be a list
                else:
                    pre_Z_collect_now = Variable(torch.zeros(2, cfg.Z_NUM))
                    pre_Z_collect_now[0, :] = pre_Z_collect[i][1, :]
                    pre_Z_collect_now[1, :] = Variable(torch.randn(1, cfg.Z_NUM))
                    pre_Z_collect = pre_Z_collect_now

                self.slerp_theta = math.acos(torch.dot(pre_Z_collect[0], 
                    pre_Z_collect[1]).data[0]/(pre_Z_collect[0].norm(p=2).\
                            data[0] * pre_Z_collect[1].norm(p=2).data[0])) 
            else:
                pre_Z_collect_now = pre_Z_collect

            file_names = data[-1]
            data = data[:-1]
            if data[2].dim() > 2: data[2] = get_edges(data[2]);
            real_imgs, mask_tensor, edge_tensor = var_and_cuda(data)
            if edge_tensor.dim() == 2: edge_tensor = None;

            img_stamp = [x + ex_stamp for x in file_names]

            self.infer_and_save_imgs(netG, real_imgs, mask_tensor, 
                    edge_tensor, None,
                    img_stamp=img_stamp, webpage=webpage, 
                    pre_Z_collect=pre_Z_collect)

            if (i+1)*data[0].size(0) == eval_num:
                break

        pbar.close()
        webpage.save()
        return pre_Z_collect

    def LPIPS_eval(self, data_loader, eval_num=100, eval_name='LPIPS_eval', 
            try_num=20):
        """
            LPIPS evaluation function

            Parameters:
            eval_num: Total number of samples needed to be evaluated
            eval_name: Evaluation name for different kinds of evaluation
            try_num: How many times to run the evaluation on the whole
        """

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        netG, _ = load_networks(self.model_dir)
        netG.eval()
        pbar = tqdm(total=try_num)

        # load the LPIPS feature extract network model
        from LPIPS.models import dist_model as dm
        dis_model = dm.DistModel()
        dis_model.initialize(model='net-lin',net='alex',use_gpu=True)

        print("\033[1;31m Doing Evaluation, Name: **%s**. \033[0m" % eval_name)

        img_collect = []
        dis_array = np.zeros(try_num)
        for j in range(try_num):
            dis_accu = 0
            pbar.update(1)
            for i, data in enumerate(data_loader):
                if i > eval_num:
                    break

                data = data[:-1] # exclude the filename
                if data[2].dim() > 2:
                    data[2] = get_edges(data[2])
                real_imgs, mask_tensor, edge_tensor = var_and_cuda(data)
                if edge_tensor.dim() == 2:
                    edge_tensor = None

                inputs = (mask_tensor, edge_tensor, None)
                for _ in range(cfg.SAMPLE_NUM):
                    while len(img_collect) != 2:
                        img_collect.append(netG(*inputs)[0].data)
                    d = dis_model.forward(*img_collect)
                    dis_accu += d.sum()
                    img_collect = []

            dis_array[j] = dis_accu / (cfg.SAMPLE_NUM * eval_num)

        pbar.close()
        output_str = '''For total %d pairs of random samples
        from %d condition, trial num %d, the LPIPS score is %.4f, std %.4f
                .\n''' % (cfg.SAMPLE_NUM * eval_num, eval_num, 
                    try_num, dis_array.mean(), dis_array.std())

        print('''\033[1;31m %s \033[0m''' % output_str)

        with open(os.path.join(self.output_dir, 'LPIPS_scores.txt'), 'a') as f:
            date_str = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')
            f.write('%s, %s' % (date_str, output_str))

        return dis_array.mean()

    def infer_and_save_imgs(self, netG, real_imgs, mask_tensor, 
            edge_tensor, epoch, 
            img_stamp=None, webpage=None, pre_Z_collect=None):
        """
            Infer and save images
            'mask_img' here represents the input condition
        """

        if cfg.TRAIN.FLAG:
            real_imgs, mask_tensor, edge_tensor, mask_tensor_to_save, \
                    edge_tensor_to_save = \
                    self.__split_tensor(real_imgs, mask_tensor, edge_tensor)
        else:
            mask_tensor_to_save, edge_tensor_to_save = mask_tensor, edge_tensor

        # If no extra image stamp, using the epoch as the stamp
        if img_stamp is None: img_stamp = epoch

        batch_size = int(real_imgs.size(0))

        # Initialize the image collection
        image_collect = [None, ] * batch_size
        for j in range(batch_size): image_collect[j] = [];

        for i in range(cfg.SAMPLE_NUM):
            # Prepare preset random latent vector if required
            if pre_Z_collect is None: pre_Z = None;
            else:
                if pre_Z_collect.size(0) == 2: # to do linear interpolation
                    sample_miu = i / float(cfg.SAMPLE_NUM)
                    pre_Z = (math.sin((1 - sample_miu) * self.slerp_theta)/\
                            math.sin(self.slerp_theta)) * pre_Z_collect[0] + \
                            (math.sin(sample_miu * self.slerp_theta)/\
                            math.sin(self.slerp_theta))  * pre_Z_collect[1]

                pre_Z = pre_Z.unsqueeze(dim=0)

            inputs = (mask_tensor, edge_tensor, pre_Z)

            gen_imgs, _ = self._parallel_wrapper(netG, inputs)

            for j in range(batch_size):
                image_collect[j].append(gen_imgs.data.cpu()[j].unsqueeze(dim=0))

        gen_imgs = [torch.cat(x, dim=0) for x in image_collect]

        for j, single_gen_imgs in enumerate(gen_imgs):
            image_data_input = OrderedDict(
                    [
                        ('real_img', real_imgs[j].unsqueeze(dim=0)),
                        ('gen_img', single_gen_imgs),
                        ('mask_img', mask_tensor_to_save[j].unsqueeze(dim=0)),
                        ]
                    )

            if edge_tensor is not None:
                image_data_input['edge_img'] =\
                        edge_tensor_to_save[0].unsqueeze(dim=0)

            if type(img_stamp) == list: img_stamp_now = img_stamp[j];
            elif type(img_stamp) == int: img_stamp_now = img_stamp;

            output_grids, img_names \
                    = save_img_results(image_data_input, img_stamp_now,
                    self.image_dir, nrow=cfg.SAMPLE_NUM)

            # Save inferred results on the webpage
            if webpage is not None and self.epoch_compare != img_stamp_now:
                webpage.add_images(img_names, img_names, img_names, 
                        [256, 256 * cfg.SAMPLE_NUM, 256])
                webpage.save()
                self.epoch_compare = img_stamp_now

        if cfg.TRAIN.FLAG:
            image_collection = OrderedDict([
                    ('real_image', output_grids[0]),
                    ('gen_image', output_grids[1]),
                    ('mask_image', output_grids[2]),
                    ])

            if len(output_grids) == 4:
                image_collection['edge_image'] = output_grids[3]

            for name, img in image_collection.items():
                self.summary_writer.add_image(name, img, epoch)

    def errE_calculate(self, netG, netD, inputs, gen_imgs, fake_digits,
            z_var, dis_mask_tensor):
        _eps = 1.0e-5
        if cfg.NOISE_EXAG:
            # Addtional feedforwarrd path
            gen_imgs_eg, z_var_eg = self._parallel_wrapper(netG, inputs)

            # Directly measure the difference of output images in 
            #     pixel-wise or using Discriminator's internal features
            if cfg.FEAT_DIFF:
                fake_digits_eg = netD(gen_imgs_eg, dis_mask_tensor)
                batch_wise_imgs_l1 = compute_digits_difference(
                        fake_digits_eg, fake_digits)
            else:
                gen_imgs_feat = gen_imgs
                gen_imgs_eg_feat = gen_imgs_eg

                batch_wise_imgs_l1 = F.l1_loss(gen_imgs_feat, 
                        gen_imgs_eg_feat.detach(), reduce=False).\
                        sum(dim=1).sum(dim=1).sum(dim=1)

                batch_wise_imgs_l1 = batch_wise_imgs_l1\
                    / (3 * cfg.IMSIZE ** 2 * cfg.IM_RATIO)

            # Measure the difference of latent vector
            batch_wise_z_l1 = F.l1_loss(
                    z_var.detach(), z_var_eg.detach(), 
                    reduce=False).\
                    sum(dim=1) / cfg.Z_NUM

            z_diff = batch_wise_z_l1
            img_diff = batch_wise_imgs_l1
                
            temp_errNE = - (img_diff / (z_diff + _eps)).mean()

            # Apply possible margin operation
            # This is not applied in default setting
            if cfg.NE_MARGIN:
                errNE = torch.clamp(temp_errNE, max=cfg.NE_MARGIN_VALUE).mean()
            else:
                errNE = temp_errNE.mean()

        else: 
            errNE = torch.zeros(1);
            if cfg.CUDA: errNE = errNE.cuda()

        return errNE, gen_imgs_eg, z_var_eg

    def get_optimizer(self, netG, netD, init_gen_lr, init_dis_lr):
        params_G = [param for param in netG.parameters() if param.requires_grad]
        optimizerG = optim.Adam(params_G,
                                lr=init_gen_lr,
                                betas=(0.5, 0.999))
        params_D = [param for param in netD.parameters() if param.requires_grad]
        optimizerD = \
            optim.Adam(params_D,
                       lr=init_dis_lr, betas=(0.5, 0.999))

        return optimizerG, optimizerD

    def __split_tensor(self, real_imgs, mask_tensor, edge_tensor):
        # only take the first batch to do inference
        mask_tensor = mask_tensor[0].unsqueeze(dim=0)
        real_imgs = real_imgs[0].unsqueeze(dim=0)
        mask_tensor_to_save = mask_tensor
        if edge_tensor is not None:
            edge_tensor = edge_tensor[0].unsqueeze(dim=0)
            edge_tensor_to_save = edge_tensor
        else: edge_tensor_to_save = None

        return real_imgs, mask_tensor, edge_tensor, mask_tensor_to_save, \
                edge_tensor_to_save

    def __set_directory(self, output_dir):
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        self.log_dir = os.path.join(output_dir, 'Log')
        self.output_dir = output_dir
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
        mkdir_p(self.log_dir)

    def _parallel_wrapper(self, net, inputs):
        if cfg.CUDA:
            return nn.parallel.data_parallel(net, inputs, self.gpus)
        else:
            return net(*inputs)

    def __change_dir(self):
        output_dir = os.path.join(self.output_dir, 'Higher')
        self.__set_directory(output_dir)
