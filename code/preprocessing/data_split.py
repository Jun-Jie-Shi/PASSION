import  os
import random


def writ_save(train_val_list:list,train_or_val_or_test :str):
    with open(train_or_val_or_test,"w") as f:
        for i in train_val_list:
            i_ = i + "\n"
            f.write(i_)

def split_data(image_path:str):
    new_image_name_list = []
    for new_image_name in os.listdir(image_path):
        new_image_name_list.append(new_image_name)

    length = len(new_image_name_list)
    random.shuffle(new_image_name_list)
    nval = int(0.1 * length)    ## set val ratio
    ntest = int(0.2 * length)   ## set test ratio
    test_list = new_image_name_list[:ntest]
    val_list = new_image_name_list[ntest:ntest+nval]
    train_list = new_image_name_list[ntest+nval:]

    return train_list,val_list,test_list


if __name__ == "__main__":
    currentdirPath = os.path.dirname(os.path.abspath(__file__))
    relativePath = '../../datasets'
    datarootPath = os.path.abspath(os.path.join(currentdirPath,relativePath))
    ## Note: or directly set datarootPath as your data-saving path (absolute root)
    src_path = os.path.join(datarootPath, 'BraTS/BRATS2020_Training_Data')
    tar_path = os.path.join(datarootPath, 'BraTS/BRATS2020_Training_none_npy')

    train_list, val_list, test_list = split_data(src_path)
    val_list.sort(key=None, reverse=False)
    test_list.sort(key=None, reverse=False)
    train_list.sort(key=None, reverse=False)
    writ_save(val_list, tar_path+"/val.txt")
    print('val save ok!')
    writ_save(test_list, tar_path+"/test.txt")
    print('test save ok!')
    writ_save(train_list, tar_path+"/train.txt")
    print('train save ok!')