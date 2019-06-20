#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:00:59 2019

@author: intern(Sayan Chatterjee)
"""

import xml.etree.ElementTree as ET 
import os
import os.path
from PIL import Image
import numpy as np
import skimage.io as io

MAX_WIDTH = 512

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


LABELS = CIRCUIT_LABELS    # change this value depending on the dataset



def prepare(mypath, output_data_path, image_count):
    out_path = os.path.join(output_data_path, "output")
    in_path = os.path.join(output_data_path, "input")
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    if not os.path.exists(in_path):
        os.makedirs(in_path)


    #Resizing the image
    for file in os.listdir(mypath):
        if(file.endswith(".tiff")):
            image_count = image_count + 1
            imgpath = os.path.join(mypath, file)

            with Image.open(imgpath) as image: 
                img_width, img_height = image.size

                multiplier = MAX_WIDTH / img_width

                WIDTH = int(img_width * multiplier)
                HEIGHT = int(img_height * multiplier)
                #print("W", WIDTH, "H", HEIGHT)
                image = image.resize((WIDTH, HEIGHT), Image.BILINEAR)
                print("image size", image.size)
                image.save(os.path.join(in_path, (str(image_count) + ".png")))
                image.transpose(Image.ROTATE_90).save(os.path.join(in_path, (str(image_count) + "_90.png")))
                image.transpose(Image.ROTATE_180).save(os.path.join(in_path, (str(image_count) + "_180.png")))
                image.transpose(Image.ROTATE_270).save(os.path.join(in_path, (str(image_count) + "_270.png")))

                WIDTH, HEIGHT = image.size
                #print("W", WIDTH, "H", HEIGHT)
                m = np.zeros((HEIGHT, WIDTH))   # to store output data
                
                pic = image
                pix = np.array(pic.getdata()).reshape(pic.size[1], pic.size[0])
                #print("pix", pix.shape)
                data = list(tuple(pixel) for pixel in pix)
                pic.putdata(data)

                #print("m shape:", m.shape)
                #print("W", WIDTH, "H", HEIGHT)
                for i in range(HEIGHT):
                    for j in range(WIDTH):
                        #print(i, j)
                        if(pix[i][j]!=255):
                            m[i][j] = LABELS["line"]

                #print(np.unique(m))

                xml_path = file.split(".")[0] + ".xml"
                xml_path = os.path.join(mypath, xml_path)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for r in root.iter('gom.std.OSymbol'):
                    x0 = int(float(r.get("x0")) * multiplier)
                    y0 = int(float(r.get("y0")) * multiplier)
                    x1 = int(float(r.get("x1")) * multiplier)
                    y1 = int(float(r.get("y1")) * multiplier)
                    label = r.get("label")
                    
                    value = LABELS[label]
                    print(value)
                    
                    for j in range(x0,(x1+1)):
                        for i in range(y0,(y1+1)):
                            if(m[i][j] == 1):
                                m[i][j] = value
                                
                output_image = os.path.join(out_path, (str(image_count) + ".png"))
                io.imsave(output_image, m.astype(np.uint8))
                
                tmp_img = Image.open(output_image)
                output_image = os.path.join(out_path, (str(image_count) + "_90.png"))
                tmp_img.transpose(Image.ROTATE_90).save(output_image)
                output_image = os.path.join(out_path, (str(image_count) + "_180.png"))
                tmp_img.transpose(Image.ROTATE_90).save(output_image)
                output_image = os.path.join(out_path, (str(image_count) + "_270.png"))
                tmp_img.transpose(Image.ROTATE_90).save(output_image)
                
                print("Image saved:", output_image)
    return image_count


if __name__ == "__main__":
    mypath = '/home/intern/Codes/Data/data/symbols/floorplans/Unzipped/floorplans16-01/floorplans16-01/'
    prepare(mypath)

            

