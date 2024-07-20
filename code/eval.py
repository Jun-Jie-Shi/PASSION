#coding=utf-8
import argparse
import logging
import os
import random
import time
from collections import OrderedDict
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# import torch.optim

from data.data_utils import init_fn
from data.datasets_nii import (Brats_loadall_test_nii,
                               Brats_loadall_val_nii)
from data.transforms import *

from models import rfnet_passion, mmformer_passion, m2ftrans_passion

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import Parser, criterions
from utils.predict import AverageMeter, test_dice_hd95_softmax
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from utils.parser import setup


parser = argparse.ArgumentParser()

parser.add_argument('--model', default='fusiontrans', type=str)
parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
# parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--dataname', default='BraTS/BRATS2020', type=str)
parser.add_argument('--datapath', default='BraTS/BRATS2020_Training_none_npy', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--mask_type', default='idt', type=str)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1037, type=int)
parser.add_argument('--needvalid', default=False, type=bool)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'testing')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

currentdirPath = os.path.dirname(__file__)
relativePath = '../datasets'
datarootPath = os.path.abspath(os.path.join(currentdirPath,relativePath))
#### Note: or directly set datarootPath as your data-saving path (absolute root):
# datarootPath = 'your data-saving path root'
dataPath = os.path.abspath(os.path.join(datarootPath,args.datapath))

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]

# masks_test = [[True, False, False, False]]
# mask_name = ['flair']
masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

masks_valid = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
# t1,t1cet1,flairticet2,flairt1cet1t2
masks_valid_torch = torch.from_numpy(np.array(masks_valid))
masks_valid_array = np.array(masks_valid)
masks_all = [True, True, True, True]
masks_all_torch = torch.from_numpy(np.array(masks_all))

mask_name_valid = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_valid_torch.int())

def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BraTS/BRATS2021', 'BraTS/BRATS2020', 'BraTS/BRATS2018']:
        num_cls = 4
    else:
        print ('dataset is error')
        exit(0)

    if args.model == 'm2ftrans_passion':
        model = m2ftrans_passion.Model(num_cls=num_cls)
    elif args.model == 'rfnet_passion':
        model = rfnet_passion.Model(num_cls=num_cls)
    elif args.model == 'mmformer_passion':
        model = mmformer_passion.Model(num_cls=num_cls)

    print (model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs, warmup=args.region_fusion_start_epoch, mode='warmuppoly')
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.AdamW(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
        ####BRATS2020
    if args.dataname == 'BraTS/BRATS2020':
        test_file = os.path.join(dataPath, 'test.txt')
    elif args.dataname == 'BraTS/BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        test_file = os.path.join(dataPath, 'test1.txt')
    elif args.dataname == 'BraTS/BRATS2021':
        ####BRATS2021
        test_file = os.path.join(dataPath, 'test.txt')

    logging.info(str(args))

    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=dataPath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    #########Evaluate
    ##########Evaluate last epoch
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('last epoch: {}'.format(checkpoint['epoch']+1))
        model.load_state_dict(checkpoint['state_dict'])
        test_dice_score = AverageMeter()
        test_hd95_score = AverageMeter()
        csv_name = os.path.join(ckpts, '{}.csv'.format(args.model))
        with torch.no_grad():
            logging.info('###########test last epoch model###########')
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow(['WT Dice', 'TC Dice', 'ET Dice','ETPro Dice', 'WT HD95', 'TC HD95', 'ET HD95' 'ETPro HD95'])
            file.close()
            for i, mask in enumerate(masks_test[::-1]):
                logging.info('{}'.format(mask_name[::-1][i]))
                file = open(csv_name, "a+")
                csv_writer = csv.writer(file)
                csv_writer.writerow([mask_name[::-1][i]])
                file.close()
                dice_score, hd95_score = test_dice_hd95_softmax(
                                test_loader,
                                model,
                                dataname = args.dataname,
                                feature_mask = mask,
                                mask_name = mask_name[::-1][i],
                                csv_name = csv_name,
                                )
                test_dice_score.update(dice_score)
                test_hd95_score.update(hd95_score)

            logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
            logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))
            exit(0)

if __name__ == '__main__':
    main()
