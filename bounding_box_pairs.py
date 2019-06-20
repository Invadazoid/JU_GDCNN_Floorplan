#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:51:56 2019

@author: prasenjit
"""

import numpy as np
from PIL import Image


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

LABELS = dict(map(reversed, FLOORPLAN_LABELS.items()))


def histogram(np_img3d, class_no, threshold=1):
    print("class:", LABELS[class_no]))
    h_arr = np.zeros(np_img3d.shape[1])
    for i in range(h_arr.shape[0]):
        for j in range(np_img3d.shape[2]):
            if(np_img3d[class_no][i][j] == 1):
                h_arr[i] = h_arr[i] + 1

    v_arr = np.zeros(np_img3d.shape[2])
    for i in range(v_arr.shape[0]):
        for j in range(np_img3d.shape[1]):
            if(np_img3d[class_no][j][i] == 1):
                v_arr[i] = v_arr[i] + 1

    
    horiz_changes = []
    vertical_changes = []
    for i in range(h_arr.shape[0]-1):
        if(h_arr[i] > threshold and h_arr[i+1] < threshold):  # new object ends
            horiz_changes.append(i)
        if(h_arr[i] < threshold and h_arr[i+1] > threshold):  # new object starts
            horiz_changes.append(i+1)
    
    for i in range(v_arr.shape[0]-1):
        if(v_arr[i] > threshold and v_arr[i+1] < threshold):  # new object ends
            vertical_changes.append(i)
        if(v_arr[i] < threshold and v_arr[i+1] > threshold):  # new object starts
            vertical_changes.append(i+1)

    bounding_box = []
    for i in range(len(horiz_changes)-1):
        for j in range(len(vertical_changes)-1):
            x0 = horiz_changes[i]
            x1 = horiz_changes[i+1]
            y0 = vertical_changes[j]
            y1 = vertical_changes[j+1]
            sum = 0
            for x in range(x0, x1):
                for y in range(y0, y1):
                   if(np_img3d[class_no][x][y] == 1):
                       sum = sum + 1
            if(sum > 50):
                box = {
                        "x0" : x0,
                        "x1" : x1,
                        "y0" : y0,
                        "y1" : y1
                        }
                bounding_box.append(box)
        
    print("Bounding box:", bounding_box)
    
    np_img2d = np.zeros((np_img3d.shape[1], np_img3d.shape[2]))
    for i in range(np_img3d.shape[1]):
        for j in range(np_img3d.shape[2]):
            np_img2d[i][j] = np_img3d[class_no][i][j]
    
    for box in bounding_box:    # for drawing bounding box
        x0 = box["x0"]
        x1 = box["x1"]
        y0 = box["y0"]
        y1 = box["y1"]
        while not (x0 == x1):
            while not (y0 == y1):
                np_img2d[x0][y0] = 1
                y0 = y0 + 1
            x0 = x0 + 1
            y0 = box["y0"]
        
    img = Image.fromarray(np_img2d*255)
    img.show()
    
        


def load_img(path):
    img = Image.open(path)
    np_img2d = np.array(img)
    shape = (np_img2d.max()+1, np_img2d.shape[0], np_img2d.shape[1])
    np_img3d = np.zeros(shape)
    for i in range(np_img2d.shape[0]):
        for j in range(np_img2d.shape[1]):
            obj_type = np_img2d[i][j]
            np_img3d[obj_type][i][j] = 1
    
    
    im = Image.fromarray(np_img3d[13,:,:]*255)
    im.show()

    histogram(np_img3d, class_no=13, threshold=1)

    
if __name__ == "__main__":
    img_path = "./img.png"
    load_img(img_path)
