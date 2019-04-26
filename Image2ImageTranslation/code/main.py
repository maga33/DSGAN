from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch

import os
import glob
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz
import json
import numpy as np
from easydict import EasyDict as edict
import shutil

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from trainer import GANTrainer

from miscc.choose_dataset import choose_dataset
from miscc.config import cfg, cfg_from_file, _merge_a_into_b
from miscc.utils import mkdir_p, find_latest_model_file, cfg_args_exchange,\
        parse_args
from FID.fid_score import fid_scores



if __name__ == "__main__":
    args = parse_args()

    # Load pre-defined file from yml file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # Synchronize the parameteres in cfg and args
    cfg_args_exchange(cfg, args) 

    print('Using config:')
    pprint.pprint(cfg)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    # Set the output directory, which will hold all the files and checkpoints
    if args.output_dir != '':
        output_dir = args.output_dir
    elif cfg.OUTPUT_DIR == '':
        output_dir = '../output/%s_TEXT_%s' % (cfg.DATASET_NAME, timestamp)
    else:
        output_dir = os.path.join('../output/', cfg.OUTPUT_DIR)

    #--------------------------------------------------
    # Load existing config.yml file or saving it
    #--------------------------------------------------
    config_file_path = os.path.join(output_dir, 'config.json')
    if os.path.isfile(config_file_path):
        print("\033[1;31m The specified folder already exists a configuration \n\
                file, we will update configuration using that. Please check \n\
                if any desired value has been changed.\033[0m")
        with open(config_file_path, 'r') as f:
            pre_cfg = edict(json.load(f))
        
        _merge_a_into_b(pre_cfg, cfg)
        pprint.pprint(cfg)

    else:
        mkdir_p(output_dir)
        with open(config_file_path, 'w') as f:
            json.dump(dict(cfg), f, indent=4)

    if args.no_cuda: cfg.CUDA = False;
    num_gpu = len(cfg.GPU_ID.split(','))

    #--------------------------------------------------
    # Update cfg flags after merge the vars from file
    #--------------------------------------------------
    if args.debug_flag: cfg.DEBUG_FLAG = True;
    if args.inter_eval or args.FID_eval or args.LPIPS_eval: args.eval = True;
    cfg.NET_G = args.net_G

    #--------------------------------------------------
    # Change some settings according to train/eval
    #--------------------------------------------------
    if args.eval: 
        cfg.TRAIN.FLAG = False
        split = 'val'
        cfg.TRAIN.BATCH_SIZE = batch_size = args.batch_size
        cfg.SAMPLE_NUM = args.sample_num
        shuffle_flag = False
        if args.LPIPS_eval: shuffle_flag=True;
        
        """
         Evaluate on all checkpoints or only on certain one
         If args.net_G is specified, then evaluate on that checkpoint, 
           other wise, evaluate on the newest checkpoint.
        """
        if not args.eval_all:
            net_G_names = [
                   os.path.basename(find_latest_model_file(
                       os.path.join(output_dir, 'Model'), cfg, 
                       keyword='netG')),]
        else:
            model_dir = os.path.join(output_dir, 'Model')
            net_G_temp = glob.glob(os.path.join(model_dir, 'netG_*'))
            net_G_temp.sort(key=os.path.getmtime, reverse=True)
            net_G_names = [os.path.basename(x) for x in net_G_temp]

            if len(net_G_names) >= args.epoch_to_eval:
                net_G_names = net_G_names[:args.epoch_to_eval]

        mkdir_p(os.path.join(output_dir, 'model_reserve'))
    else:
        split = 'train'
        batch_size = cfg.TRAIN.BATCH_SIZE * num_gpu
        shuffle_flag = True

    Dataset = choose_dataset(cfg.DATASET_NAME)
    dataset = Dataset(cfg.DATA_DIR, split, imsize=cfg.IMSIZE)

    # Note the batchsize setting is here
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        drop_last=True, shuffle=shuffle_flag, num_workers=int(cfg.WORKERS))

    # Initialize the main class which includes the training and evaluation
    algo = GANTrainer(output_dir, cfg_path=args.cfg_file)

    if cfg.TRAIN.FLAG:
        algo.train(dataloader)

    elif args.eval:
        date_str = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')

        if args.FID_eval:
            '''Do FID evaluations.'''
            f = open(os.path.join(output_dir, 'all_FID_eval.txt'), 'a')
            for net_G_name in net_G_names:
                cfg.NET_G = net_G_name
                algo.sample(dataloader, eval_name='eval',
                        eval_num=args.eval_num)
                fid_score_now = \
                    fid_scores(output_dir, cfg, sample_num=args.sample_num, 
                    gen_images_path=args.gen_paths, loop=True)

                f.write('%s, %s, %.4f\n' % (date_str, net_G_name, fid_score_now))

            f.close()

            # Save the best FID score model
            with open(os.path.join(output_dir, 'all_FID_eval.txt'), 'r') as f:
                all_lines = f.readlines()
                score_array =\
                        np.asarry([float(line.strip('\n').split(', ')[-1]) \
                        for line in all_lines])
                if fid_score_now == score_array.min():
                    print("save the best FID score model, the score is %.4f" % \
                            fid_score_now)

                    net_G_name = all_lines[score_array.argmin()].split(', ')[1]
                    shutil.copy(os.path.join(output_dir, 'Model', net_G_name), 
                            os.path.join(output_dir, 'model_reserve'))


        elif args.inter_eval:
            '''Choose two random variables and do interpolation'''
            algo.sample(dataloader, eval_name='inter_eval', eval_num=args.eval_num)
        elif args.LPIPS_eval:
            '''Do LPIPS evaluation'''
            f = open(os.path.join(output_dir, 'all_LPIPS_eval.txt'), 'a')
            mean_collect = np.zeros(len(net_G_names))
            for i, net_G_name in enumerate(net_G_names):
                cfg.NET_G = net_G_name
                LPIPS_mean = algo.LPIPS_eval(dataloader, eval_name='LPIPS_eval', 
                        eval_num=args.eval_num, try_num=args.try_num)
                mean_collect[i] = LPIPS_mean
                f.write('%s, %s, %.4f\n' % (date_str, net_G_name, LPIPS_mean))

            f.close()
        else:
            '''Randomly sample for each evaluation case '''
            print("each input will sample %d output" % cfg.SAMPLE_NUM)
            algo.sample(dataloader, eval_name='eval', eval_num=args.eval_num)

