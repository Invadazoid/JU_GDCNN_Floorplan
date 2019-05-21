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

#Creating unique list and appending with values  
l = []
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []

#Selecting the XML files
mypath = '/home/intern/Codes/Data/floorplans16-01/'
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
#Writing and saving unique and sorted list to file
with open('Furniture Names.txt', 'w') as f:
    for item in l1:
        f.write("%s\n" % item)  
        
imgpath = "/home/intern/Codes/Data/floorplans16-01/file_0.tiff"
img  = Image.open(imgpath) 
with Image.open(imgpath) as image: 
    width, height = image.size  
print(width,"",height)

for file in listdir(mypath):
    if file.endswith(".xml"):
        for r in root.iter('gom.std.OSymbol'):
            x0 = r.get("x0")
            x1 = r.get("y0")
            x2 = r.get("x1")
            x3 = r.get("y1")



 