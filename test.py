#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:00:59 2019

@author: intern(Sayan Chatterjee)
"""

import csv 
import requests 
import xml.etree.ElementTree as ET 
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np


#Creating unique list and appending with values  
l = []
l1 = []
c = []

#Selecting the XML files
mypath = '/home/intern/Codes/Data/data/symbols/floorplans/Unzipped/floorplans16-01/floorplans16-01/'
for file in listdir(mypath):
    if file.endswith(".xml"):
        filepath = mypath + file
       
#Creating instance object for ET       
        tree = ET.parse(filepath)  
        root = tree.getroot()
       
        for r in root.iter('gom.std.OSymbol'):
            l.append(r.get("label"))
    
        for x in l: 
            if x not in l1: 
                l1.append(x)
l1.sort()
print(l1)

for i in range(len(l1)):
    c.append(i)

#Writing and saving unique and sorted list to file
with open('Furniture Names.txt', 'w') as f:
    for item in l1:
        f.write("%s\n" % item)  

#Test code to extract and open image        
imgpath = "/home/intern/Codes/Data/data/symbols/floorplans/Unzipped/floorplans16-01/floorplans16-01/file_0.tiff"
img  = Image.open(imgpath) 
with Image.open(imgpath) as image: 
    width, height = image.size  
print(width,"",height)

#Resizing the image
for file in listdir(mypath):
    if file.endswith(".xml"):
        for r in root.iter('gom.std.OSymbol'):
            x0 = int(float(r.get("x0"))/13.232421875)
            y0 = int(float(r.get("y0"))/13.2314814815)
            x1 = int(float(r.get("x1"))/13.232421875)
            y1 = int(float(r.get("y1"))/13.2314814815)
            print("x0", x0*13.232421875)
#Creating matrix of appropriate size to list 0's and 1's        
m = np.zeros((512,216))

pic = Image.open(imgpath)
pix = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1])
data = list(tuple(pixel) for pixel in pix)
pic.putdata(data)

for i in range(x0,(x1+1)):
    for j in range(y0,(y1+1)):
        if(pix[i][j]!=255):
            m[i][j] = 1

for i in range(x0,(x1+1)):
    for j in range(y0,(y1+1)):
        if(m[i][j] == 1):
            m[i][j] = c[i]



            

 