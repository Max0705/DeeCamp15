# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:35:45 2018

@author: kongz
"""

import os
import imageio
import cv2
import numpy as np


def img2gif(path, suffix='jpg', outfile='outfile.gif', length_limit=0):
    l = len(os.listdir(path))
    files = ['img_' + str(i) + '.' + suffix for i in range(l)]
    if length_limit > 0:
        files = files[:length_limit]
    
    shape = cv2.imread(os.path.join(path, files[0])).shape
    
    with imageio.get_writer(outfile, mode='I') as writer:    
        for file in files:
            img = cv2.imread(os.path.join(path, file))
            if img is None or (img.shape != shape):
                continue
            img = np.array([[x[::-1] for x in y] for y in img])
            print(file)
            writer.append_data(img)        
            
    print('GIF generated')


# name of images: img_0.jpg, img_1.jpg, ..., etc
img2gif('gif/input/car', 
        suffix='jpg',
        outfile='gif/output/output.gif', 
        length_limit=30)
