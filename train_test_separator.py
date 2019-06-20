#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:59:31 2019

@author: prasenjit
"""

import os
import os.path
import random
import shutil


IN_DATA_PATH = os.path.join(os.getcwd(), "floor_plan_data")
OUT_DATA_PATH = os.path.join(os.getcwd(), "data")

if not os.path.isdir(OUT_DATA_PATH):
    os.makedirs(OUT_DATA_PATH)

TRAIN_DATA_PATH = os.path.join(OUT_DATA_PATH, "train")
VALIDATION_DATA_PATH = os.path.join(OUT_DATA_PATH, "valid")
TEST_DATA_PATH = os.path.join(OUT_DATA_PATH, "test")

if not os.path.isdir(TRAIN_DATA_PATH):
    os.makedirs(TRAIN_DATA_PATH)
    
if not os.path.isdir(VALIDATION_DATA_PATH):
    os.makedirs(VALIDATION_DATA_PATH)
    
if not os.path.isdir(TEST_DATA_PATH):
    os.makedirs(TEST_DATA_PATH)
    
pic_list = os.listdir(os.path.join(IN_DATA_PATH, "input"))
random.shuffle(pic_list)
print(pic_list)


total_pics = len(pic_list)
print(total_pics)

n_train = int(0.6 * total_pics)
n_valid = int(0.2 * total_pics)
n_test = int(0.2 * total_pics)

print(n_train, n_valid, n_test)

train_pic_list = pic_list[:n_train]
valid_pic_list = pic_list[n_train:(n_train+n_valid)]
test_pic_list = pic_list[(n_train + n_valid): ]

print()
print("train", train_pic_list)
print()
print("valid", valid_pic_list)
print()
print("test", test_pic_list)

def copy(src_list, dest_path):
    dest_inp = os.path.join(dest_path, "input")
    dest_out = os.path.join(dest_path, "output")
    os.makedirs(dest_inp)
    os.makedirs(dest_out)
    input_image_folder = os.path.join(IN_DATA_PATH, "input")
    output_image_folder = os.path.join(IN_DATA_PATH, "output")
    for i in src_list:
        in_img = os.path.join(input_image_folder, i)
        out_img = os.path.join(output_image_folder, i)
        
        shutil.copyfile(in_img, os.path.join(dest_inp, i))
        shutil.copyfile(out_img, os.path.join(dest_out, i))

copy(train_pic_list, TRAIN_DATA_PATH)
copy(valid_pic_list, VALIDATION_DATA_PATH)
copy(test_pic_list, TEST_DATA_PATH)

