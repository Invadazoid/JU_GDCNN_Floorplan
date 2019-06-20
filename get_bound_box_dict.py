#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:51:56 2019

@author: prasenjit
"""

import numpy as np
from PIL import Image
from skimage import measure
import skimage.io as io

FLOORPLAN_LABELS = {
        "background":0,
        "line":1,
        "armchair":2,
        "bed":3,
        "door1":4,
        "door2":5,
        "sink1":6,
        "sink2":7,
        "sink3":8,
        "sink4":9,
        "sofa1":10,
        "sofa2":11,
        "table1":12,
        "table2":13,
        "table3":14,
        "tub":15,
        "window1":16,
        "window2":17
        }


CIRCUIT_LABELS = {
        "background":0,
        "line":1,
        "battery1":2,
        "battery2":3,
        "capacitor1":4,
        "capacitor2":5,
        "core-air":6,
        "core-hiron":7,
        "core-iron":8,
        "diode1":9,
        "diode2":10,
        "diode3":11,
        "diodephoto":12,
        "earth":13,
        "gate-ampli":14,
        "outlet":15,
        "relay":16,
        "resistor":17,
        "transistor-mosfetn":18,
        "transistor-mosfetp":19,
        "transistor-npn":20,
        "transistor-pnp":21,
        "unspecified":22
}

#LABELS = dict(map(reversed, FLOORPLAN_LABELS.items()))


def bbox(np_img3d, class_no, threshold=1):
    label = measure.label(np_img3d[class_no,:,:])
    bounding_box = []
    for i in range(label.max()):
        props = measure.regionprops((label == (i+1)).astype(np.uint8))
        for prop in props:
            #print(i, prop.bbox)
            box = prop.bbox
            bounding_box.append({
                    "x0" : box[0],
                    "x1" : box[2],
                    "y0" : box[1],
                    "y1" : box[3]
                    })
    return bounding_box

def bound_box(path):
    img = Image.open(path)
    np_img2d = np.array(img)
    shape = (np_img2d.max()+1, np_img2d.shape[0], np_img2d.shape[1])
    np_img3d = np.zeros(shape)
    for i in range(np_img2d.shape[0]):
        for j in range(np_img2d.shape[1]):
            obj_type = np_img2d[i][j]
            np_img3d[obj_type][i][j] = 1

    bounding_box = {}
    for i in range(2, np_img2d.max()+1):    # range starts from 2 as 0 & 1 corresponds to lines and background
        #im = Image.fromarray((np_img3d[i,:,:]*255).astype('uint8'))
        #im.save("class_" + str(i) + ".png")
        boxes = bbox(np_img3d, class_no=i, threshold=1)
        bounding_box[i] = boxes
        
    print(bounding_box)
    
if __name__ == "__main__":
    img_path = "./img.png"
    bound_box(img_path)
