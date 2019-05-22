#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:58:23 2019

@author: intern
"""

import data_prep
import os
import os.path


def prep(path):
    print("Going to path:", path)
    data_prep.prepare(path)

if __name__ == "__main__":
    base_path = "/home/intern/Codes/Data/data/symbols/floorplans/Unzipped/"
    dir_list = os.listdir(base_path)
    print(dir_list)
    for f in dir_list:
        print("fold:", f)
        path = os.path.join(base_path, f)
        if os.path.isdir(path):
            path = os.path.join(path, f)
            if(os.path.isdir(path)):
                print(path)
                prep(path)