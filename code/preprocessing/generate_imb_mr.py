import csv
import os
import numpy as np
import torch
import random
import math

currentdirPath = os.path.dirname(os.path.abspath(__file__))
relativePath = '../../datasets'
datarootPath = os.path.abspath(os.path.join(currentdirPath,relativePath))
## Note: or directly set datarootPath as your data-saving path (absolute root)
train_path = os.path.join(datarootPath, 'BraTS/BRATS2020_Training_none_npy')
train_file = os.path.join(train_path, 'train.txt')
split_path = os.path.join(datarootPath, 'BraTS/brats_split')
os.makedirs(split_path, exist_ok=True)
csv_name = os.path.join(split_path, 'Brats2020_imb_split_mr2468.csv')

p=[0.2, 0.4, 0.6, 0.8]

## set_seed
seed = 1037
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

with open(train_file, 'r') as f:
    datalist = [i.strip() for i in f.readlines()]
datalist.sort()

img_max = len(datalist)
img_num_per_cls = [0 for i in range(15)]

def imb_mr_split(p, img_max):

    ## For BraTS -- Four modalities: T1 T1c Flair T2
    t1 = np.random.rand(img_max)>p[0]
    t1c = np.random.rand(img_max)>p[1]
    flair = np.random.rand(img_max)>p[2]
    t2 = np.random.rand(img_max)>p[3]

## PS: Using np.random.rand can also generate imb_mr but finding imb_mr not as expected,
## so we chose to i.i.d.-expected split instead, as follows:
    count = 0

    tttt = int(img_max*(1-p[0])*(1-p[1])*(1-p[2])*(1-p[3]))
    tttt = tttt if tttt >0 else tttt+1
    t1[count:count+tttt]=True
    t1c[count:count+tttt]=True
    flair[count:count+tttt]=True
    t2[count:count+tttt]=True
    count += tttt

    ttft = int(img_max*(1-p[0])*(1-p[1])*(p[2])*(1-p[3]))
    ttft = ttft if ttft >0 else ttft+1
    t1[count:count+ttft]=True
    t1c[count:count+ttft]=True
    flair[count:count+ttft]=False
    t2[count:count+ttft]=True
    count += ttft

    tttf = int(img_max*(1-p[0])*(1-p[1])*(1-p[2])*(p[3]))
    tttf = tttf if tttf >0 else tttf+1
    t1[count:count+tttf]=True
    t1c[count:count+tttf]=True
    flair[count:count+tttf]=True
    t2[count:count+tttf]=False
    count += tttf

    ttff = int(img_max*(1-p[0])*(1-p[1])*(p[2])*(p[3]))
    ttff = ttff if ttff >0 else ttff+1
    t1[count:count+ttff]=True
    t1c[count:count+ttff]=True
    flair[count:count+ttff]=False
    t2[count:count+ttff]=False
    count += ttff

    tftt = int(img_max*(1-p[0])*(p[1])*(1-p[2])*(1-p[3]))
    tftt = tftt if tftt >0 else tftt+1
    t1[count:count+tftt]=True
    t1c[count:count+tftt]=False
    flair[count:count+tftt]=True
    t2[count:count+tftt]=True
    count += tftt

    tftf = int(img_max*(1-p[0])*(p[1])*(1-p[2])*(p[3]))
    tftf = tftf if tftf >0 else tftf+1
    t1[count:count+tftf]=True
    t1c[count:count+tftf]=False
    flair[count:count+tftf]=True
    t2[count:count+tftf]=False
    count += tftf

    tfft = int(img_max*(1-p[0])*(p[1])*(p[2])*(1-p[3]))
    tfft = tfft if tfft >0 else tfft+1
    t1[count:count+tfft]=True
    t1c[count:count+tfft]=False
    flair[count:count+tfft]=False
    t2[count:count+tfft]=True
    count += tfft

    tfff = int(img_max*(1-p[0])*(p[1])*(p[2])*(p[3]))
    tfff = tfff if tfff >0 else tfff+1
    t1[count:count+tfff]=True
    t1c[count:count+tfff]=False
    flair[count:count+tfff]=False
    t2[count:count+tfff]=False
    count += tfff

    fttt = int(img_max*p[0]*(1-p[1])*(1-p[2])*(1-p[3]))
    fttt = fttt if fttt >0 else fttt+1
    t1[count:count+fttt]=False
    t1c[count:count+fttt]=True
    flair[count:count+fttt]=True
    t2[count:count+fttt]=True
    count += fttt

    ftft = int(img_max*(p[0])*(1-p[1])*(p[2])*(1-p[3]))
    ftft = ftft if ftft >0 else ftft+1
    t1[count:count+ftft]=False
    t1c[count:count+ftft]=True
    flair[count:count+ftft]=False
    t2[count:count+ftft]=True
    count += ftft

    fttf = int(img_max*(p[0])*(1-p[1])*(1-p[2])*(p[3]))
    fttf = fttf if fttf >0 else fttf+1
    t1[count:count+fttf]=False
    t1c[count:count+fttf]=True
    flair[count:count+fttf]=True
    t2[count:count+fttf]=False
    count += fttf

    ftff = int(img_max*(p[0])*(1-p[1])*(p[2])*(p[3]))
    ftff = ftff if ftff >0 else ftff+1
    t1[count:count+ftff]=False
    t1c[count:count+ftff]=True
    flair[count:count+ftff]=False
    t2[count:count+ftff]=False
    count += ftff

    fftt = int(img_max*(p[0])*(p[1])*(1-p[2])*(1-p[3]))
    fftt = fftt if fftt >0 else fftt+1
    t1[count:count+fftt]=False
    t1c[count:count+fftt]=False
    flair[count:count+fftt]=True
    t2[count:count+fftt]=True
    count += fftt

    fftf = int(img_max*(p[0])*(p[1])*(1-p[2])*(p[3]))
    fftf = fftf if fftf >0 else fftf+1
    t1[count:count+fftf]=False
    t1c[count:count+fftf]=False
    flair[count:count+fftf]=True
    t2[count:count+fftf]=False
    count += fftf

    ffft = int(img_max*(p[0])*(p[1])*(p[2])*(1-p[3]))
    ffft = ffft if ffft >0 else ffft+1
    t1[count:count+ffft]=False
    t1c[count:count+ffft]=False
    flair[count:count+ffft]=False
    t2[count:count+ffft]=True
    count += ffft

    # ffff = int(img_max*(p[0])*(p[1])*(p[2])*(p[3]))
    t1[count:]=False
    t1c[count:]=False
    flair[count:]=False
    t2[count:]=False

    return t1, t1c, flair, t2

# Statistics Summary
# [t1 t1c flair t2]
# BraTS2018
# 1379 182 141 61 20
# 2468 170 121 82 41
# BraTS2020
# 8825: 46 44 189 118
# 8852: 46 44 116 190
# 5582: 113 111 42 186
# 5528: 113 111 185 42
# 2258: 179 180 111 44
# 2285: 179 180 43 113
# 2468: 184 135 90 43
# 1379: 200 156 68 23

t1, t1c, flair, t2 = imb_mr_split(p, img_max)
state = np.random.get_state()
np.random.shuffle(t1)
np.random.set_state(state)
np.random.shuffle(t1c)
np.random.set_state(state)
np.random.shuffle(flair)
np.random.set_state(state)
np.random.shuffle(t2)

index = 0
pos_index = []

## Counting and Saving Statistics of Imbalanced-MR BraTS Data

# mask_array = np.array([[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
#          [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
#          [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
#          [True, True, True, True]])
file = open(csv_name, "a+")
csv_writer = csv.writer(file)
csv_writer.writerow(['data_name', 'mask_id', 'mask', 'pos_mask_ids']) ## possible mask id (after-moddrop)
for i in range(img_max):
    while (not t1[i] and not t1c[i] and not flair[i] and not t2[i]):
        print(i)
        t1[i] = np.random.rand(1)>p[0]
        t1c[i] = np.random.rand(1)>p[1]
        flair[i] = np.random.rand(1)>p[2]
        t2[i] = np.random.rand(1)>p[3]
        print([t1[i],t1c[i],flair[i],t2[i]])
    if [flair[i],t1c[i],t1[i],t2[i]] == [False, False, True, False]:
        img_num_per_cls[0] +=1
        index = 2
        pos_index = [2]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [False, True, False, False]:
        img_num_per_cls[1] +=1
        index = 1
        pos_index = [1]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [True, False, False, False]:
        img_num_per_cls[2] +=1
        index = 3
        pos_index = [3]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [False, False, False, True]:
        img_num_per_cls[3] +=1
        index = 0
        pos_index = [0]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [False, True, True, False]:
        img_num_per_cls[4] +=1
        index = 5
        pos_index = [1,2,5]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [True, False, True, False]:
        img_num_per_cls[5] +=1
        index = 6
        pos_index = [2,3,6]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [False, False, True, True]:
        img_num_per_cls[6] +=1
        index = 7
        pos_index = [0,2,7]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [True, True, False, False]:
        img_num_per_cls[7] +=1
        index = 9
        pos_index = [1,3,9]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [False, True, False, True]:
        img_num_per_cls[8] +=1
        index = 4
        pos_index = [0,1,4]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [True, False, False, True]:
        img_num_per_cls[9] +=1
        index = 8
        pos_index = [0,3,8]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [True, True, True, False]:
        img_num_per_cls[10] +=1
        index = 10
        pos_index = [1,2,3,5,6,9,10]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [False, True, True, True]:
        img_num_per_cls[11] +=1
        index = 13
        pos_index = [0,1,2,4,5,7,13]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [True, False, True, True]:
        img_num_per_cls[12] +=1
        index = 11
        pos_index = [0,2,3,6,7,8,11]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [True, True, False, True]:
        img_num_per_cls[13] +=1
        index = 12
        pos_index = [0,1,3,4,8,9,12]
    elif [flair[i],t1c[i],t1[i],t2[i]] == [True, True, True, True]:
        img_num_per_cls[14] +=1
        index = 14
        pos_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    
    csv_writer = csv.writer(file)
    csv_writer.writerow([datalist[i],index,[flair[i],t1c[i],t1[i],t2[i]],pos_index])
file.close()
print(img_num_per_cls)
print(sum(img_num_per_cls))

img_num_per_cls = np.array(img_num_per_cls)

t1_index = [0,4,5,6,10,11,12,14]
t1c_index = [1,4,7,8,10,11,13,14]
flair_index = [2,5,7,9,10,12,13,14]
t2_index = [3,6,8,9,11,12,13,14]

t1 = img_num_per_cls[t1_index]
t1c = img_num_per_cls[t1c_index]
flair = img_num_per_cls[flair_index]
t2=img_num_per_cls[t2_index]
print([sum(t1),sum(t1c),sum(flair),sum(t2)])


