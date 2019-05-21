import os
import numpy as np
import torch
from PIL import Image
from util.progbar import progbar
import json
import pandas as pd
from itertools import groupby
from tqdm import tqdm
import cv2

category_num = 46 + 1

WIDTH = 512
HEIGHT = 512

def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    label_desc_path = os.path.join(opt.data, 'label_descriptions.json')
    label_desc = json.load(open(label_desc_path))
    # print(label_desc)
    df = pd.read_csv(open(os.path.join(opt.data, 'train.csv'), 'r'))
    img_ind_num = df.groupby("ImageId")["ClassId"].count()
    all_trainval_num = img_ind_num.count()
    split_rate = 0.75
    train_num = np.ceil(split_rate * all_trainval_num)
    train, val = processing(opt.data, img_ind_num, train_num, df)
    info = {
        "label_desc": label_desc,
        'base_dir': opt.data,
        'train': train,
        'val': val,
        'test': os.listdir(os.path.join(opt.data, 'test'))
    }
    print("************* Generating list of data ....**************")
    print("^^^^^^^^^^^^ Saved the cached file in {} ^^^^^^^^^^^^^".format(cacheFilePath))
    torch.save(info, cacheFilePath)
    return info




def make_mask_image(segment_df):
    seg_width = segment_df.at[0, "Width"]
    seg_height = segment_df.at[0, "Height"]
    seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)
    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
        pixel_list = list(map(int, encoded_pixels.split(" ")))
        for i in range(0, len(pixel_list), 2):
            start_index = pixel_list[i] - 1
            index_len = pixel_list[i+1] - 1
            # if int(class_id.split("_")[0])  == 0:
            #     print('exists zer0 class id.')
            seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])
    seg_img = seg_img.reshape((seg_height, seg_width), order='F')
    seg_img = cv2.resize(seg_img, (512, 512), interpolation=cv2.INTER_NEAREST)
    """
    seg_img_onehot = np.zeros((HEIGHT, WIDTH, category_num), dtype=np.int32)
    #seg_img_onehot = np.zeros((seg_height//ratio, seg_width//ratio, category_num), dtype=np.int32)
    # OPTIMIZE: slow
    for ind in range(HEIGHT):
        for col in range(WIDTH):
            seg_img_onehot[ind, col] = make_onehot_vec(seg_img[ind, col])
    """
    return seg_img



def make_onehot_vec(x):
    vec = np.zeros(category_num)
    vec[x] = 1
    return vec

def processing(path, img_ind_num, train_num, data_frame):
    print("=> start collecting training data....")
    train_img_dir = os.path.join(path, 'train')
    index = data_frame.index.values[0]
    train_set = []
    val_set = []

    for i, (img_name, ind_num) in tqdm(enumerate(img_ind_num.items())):
        # print(img_name, train_img_dir)
        segment_df = (data_frame.loc[index:index + ind_num - 1, :]).reset_index(drop=True)
        index += ind_num
        if segment_df["ImageId"].nunique() != 1:
            raise Exception("Index Range Error")
        seg_img = make_mask_image(segment_df)
        mask_file_path = os.path.join(train_img_dir, 'mask_'+img_name.replace('.jpg', '.png'))
        img_path = os.path.join(train_img_dir, img_name)

        cv2.imwrite(mask_file_path, seg_img)
        if i < train_num:
            train_set.append((img_path, mask_file_path))
        else:
            val_set.append((img_path, mask_file_path))

    return train_set, val_set
