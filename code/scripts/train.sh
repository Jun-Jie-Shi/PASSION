#!/bin/bash
pythonname='/home/sjj/miniconda3/envs/pytorch' ## your env-path

dataname='BraTS/BRATS2020'
datapath=${dataname}_Training_none_npy
pypath=$pythonname
cudapath='/home/sjj/anaconda3' ## your cuda-path
savepath='outputs/idt_mr2468_mmformer_passion_bs1_epoch300_lr2e-4_temp4'
imbmrpath='BraTS/brats_split/Brats2020_imb_split_mr2468.csv'
model='mmformer'
mask_type='idt'

export PATH=$cudapath/bin:$PATH
export LD_LIBRARY_PATH=$cudapath/lib:$LD_LIBRARY_PATH
PYTHON=$pypath/bin/python3.8 ## your python-path
export PATH=$pypath/include:$pypath/bin:$PATH
export LD_LIBRARY_PATH=$pypath/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0
#train:
# without pretrain
$PYTHON train.py --model $model --batch_size=1 --mask_type $mask_type --use_passion \
    --temp 4 --num_epochs 300 --lr 2e-4 --region_fusion_start_epoch 0 \
    --dataname $dataname --datapath $datapath --savepath $savepath --imbmrpath $imbmrpath