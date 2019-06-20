#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:10:35 2019

@author: prasenjit
"""

from skimage import util
from PIL import Image
import numpy as np
import os
import os.path

import shutil

def add_gaussian_noise(image_path, output_path, mean, sigma):
    img = Image.open(image_path).convert('LA')
    W, H = img.size
    
    np_img = np.array(img)[:,:,0]
    gaussian = np.random.normal(loc=mean, scale=sigma, size=(H,W))
    
    new_img = (np_img + gaussian)
    new_img = (new_img - new_img.min())/(new_img.max() - new_img.min())
    new_img = new_img*255
    
    new_img = new_img.astype(np.uint8)
    im = Image.fromarray(new_img)
    im.save(output_path)
    
    
if __name__ == "__main__":
    input_path = "/home/intern/Codes/Data/final_data/sesyd_circuit_data_ready2train/circuit_data/input/"
    output_path = "/home/intern/Codes/Data/final_data/sesyd_circuit_data_ready2train/circuit_data/output/"
    desti_input_path = "/home/intern/Codes/Data/final_data/sesyd_circuit_data_ready2train/data_n30/input/"
    desti_output_path = "/home/intern/Codes/Data/final_data/sesyd_circuit_data_ready2train/data_n30/output/"
    if not os.path.isdir(desti_input_path):
        os.makedirs(desti_input_path, mode=0o777)
    if not os.path.isdir(desti_output_path):
        os.makedirs(desti_output_path, mode=0o777)
    inp_list = os.listdir(input_path)
    out_list = os.listdir(output_path)
    for i in inp_list:
        add_gaussian_noise(image_path=os.path.join(input_path, i),
                           output_path=os.path.join(desti_input_path,i),
                           mean=0,
                           sigma=30)
    for i in out_list:
        shutil.copy(src=os.path.join(output_path, i), dst=os.path.join(desti_output_path,i))