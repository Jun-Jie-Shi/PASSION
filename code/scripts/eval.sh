#!/bin/bash
pythonname='/home/sjj/miniconda3/envs/passion' ## your env-path

dataname='BraTS/BRATS2020'
pypath=$pythonname
cudapath='/home/sjj/miniconda3' ## your cuda-path
datapath=${dataname}_Training_none_npy
savepath='outputs/eval_mmformer'
resume='your pretrained model path'
model='mmformer'

export PATH=$cudapath/bin:$PATH
export LD_LIBRARY_PATH=$cudapath/lib:$LD_LIBRARY_PATH
PYTHON=$pypath/bin/python3.8 ## your python-path
export PATH=$pypath/include:$pypath/bin:$PATH
export LD_LIBRARY_PATH=$pypath/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0

#eval:
$PYTHON eval.py --datapath $datapath --savepath $savepath --dataname $dataname --resume $resume --model $model

