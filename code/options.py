import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='mmformer', type=str, help='model name')
    parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')

    parser.add_argument('--lr', default=2e-4, type=float, help='base learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=300, type=int, help='training epochs')
    # parser.add_argument('--iter_per_epoch', default=150, type=int)
    parser.add_argument('--temp', default=4.0, type=float, help='knowledge-distillation temperature')
    parser.add_argument('--region_fusion_start_epoch', default=0, type=int, help='warm-up epochs used in rfnet')

    #### System setting
    parser.add_argument('--seed', default=1037, type=int, help='random seed')
    parser.add_argument('--gpu', type=str, default='3', help='GPU to use')

    #### Option-choice setting
    parser.add_argument('--mask_type', default='idt', type=str, help='training settings: pdt idt or idt_drop')
    parser.add_argument('--use_pretrain', action='store_true', default=False, help='whether use pretrained model')
    parser.add_argument('--use_passion', action='store_true', default=False, help='whether use passion')
    parser.add_argument('--use_valid', action='store_true', default=False, help='whether use validation')

    #### Path setting
    parser.add_argument('--dataname', default='BraTS/BRATS2020', type=str, help='stored dataset name')
    parser.add_argument('--datapath', default='BraTS/BRATS2020_Training_none_npy', type=str, help='stored dataset path (ralative to datarootPath)')
    parser.add_argument('--imbmrpath', default='BraTS/brats_split/Brats2020_imb_split_mr2468.csv', type=str, help='csv path')
    parser.add_argument('--savepath', default='outputs/idt_mr2468_mmformer_passion_bs1_epoch300_lr2e-4_temp4', type=str, help='output path')
    parser.add_argument('--resume', default=None, type=str, help='pretrained model path')

    #### Relative path setting
    currentdirPath = os.path.dirname(__file__)
    relativePath = '../datasets'
    datarootPath = os.path.abspath(os.path.join(currentdirPath,relativePath))
    #### Note: or directly set datarootPath as your data-saving path (absolute root):
    # datarootPath = 'your data-saving path (root)'
    # datarootPath = '/home/sjj/PASSION/datasets'

    args = parser.parse_args()

    args.datarootPath = datarootPath
    args.datasetPath = os.path.abspath(os.path.join(args.datarootPath,args.datapath))
    # args.datarootPath = 'your data-saving path (root)'
    # args.datasetPath = 'your dataset-saving path (absolute path)'

    #### Note: 3D Segmentation transform for BraTS
    args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
    args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    return args
