# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:04:50 2018

@author: kongz
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random


def get_line_pos(start, end, p):
    return (start[0] + int(p*(end[0]-start[0])), 
            start[1] + int(p*(end[1]-start[1])))


def split(n, low_perc=0.6):
    p = [0]
    empty = 1 - low_perc
    for index in range(n-1):
        p.append(p[-1] + (random.random()*0.4) * (1-p[-1]))
        blank = (random.random()*0.6+0.2) * empty
        p.append(p[-1] + blank)
        empty -= blank
    p.append(1)
    return p
    
    
    

def generate_line(size=512, start=(0,0), end=(511,511),
                  n_split=2, low_perc=0.6, 
                  color=(155,155,155), width=10):
    
    split_perc = split(n_split, low_perc=low_perc)
    
    for index in range(n_split):
        split_start = get_line_pos(start, end, split_perc[2*index])
        split_end = get_line_pos(start, end, split_perc[2*index+1])
        cv2.line(img, split_start, split_end, color, width)
        
    
def generate_square(size=512, lefttop=(0,0), rightdown=(511,511),
                  n_split=2, low_perc=0.6, 
                  color=(155,155,155), width=10):
    
    generate_line(size=size, 
                  start=(lefttop[0],lefttop[1]), 
                  end=(rightdown[0],lefttop[1]),
                  n_split=n_split, low_perc=low_perc, 
                  color=color, width=width)
    
    generate_line(size=size, 
                  start=(rightdown[0],lefttop[1]), 
                  end=(rightdown[0],rightdown[1]),
                  n_split=n_split, low_perc=low_perc, 
                  color=color, width=width)
    
    generate_line(size=size, 
                  start=(rightdown[0],rightdown[1]), 
                  end=(lefttop[0],rightdown[1]),
                  n_split=n_split, low_perc=low_perc, 
                  color=color, width=width)
    
    generate_line(size=size, 
                  start=(lefttop[0],rightdown[1]), 
                  end=(lefttop[0],lefttop[1]),
                  n_split=n_split, low_perc=low_perc, 
                  color=color, width=width)


def generate_triangle(size=512, top=(0,256), length=256,
                  n_split=2, low_perc=0.6, 
                  color=(155,155,155), width=10):
    
    leftdown = (top[0]+int(0.866*length), top[1]-int(length/2))
    rightdown = (top[0]+int(0.866*length), top[1]+int(length/2))

    generate_line(size=size, 
                  start=top, 
                  end=leftdown,
                  n_split=n_split, low_perc=low_perc, 
                  color=color, width=width)
    
    generate_line(size=size, 
                  start=leftdown, 
                  end=rightdown,
                  n_split=n_split, low_perc=low_perc, 
                  color=color, width=width)
    
    generate_line(size=size, 
                  start=rightdown, 
                  end=top,
                  n_split=n_split, low_perc=low_perc, 
                  color=color, width=width)
    
    
def generate_circle(size=512, center=(256,256), radius=100,
                  n_split=3, low_perc=0.6, complete=1, shift=0,
                  color=(155,155,155), width=10):
    
    split_perc = split(n_split, low_perc=low_perc)
    split_perc = [x * complete for x in split_perc]
    
    for index in range(n_split):
        degree_start = int(360 * split_perc[2*index]) + shift
        degree_end = int(360 * split_perc[2*index+1]) + shift

        cv2.ellipse(img, center, (radius, radius), 0, 
                    degree_start, degree_end, 
                    color=color, thickness=width)


def generate_circ_with_array(size=512, center=(256,256), radius=100,
                  n_split=3, low_perc=0.6, complete=1, shift=0,
                  color=(155,155,155), width=10,
                  arr_len = 100, arr_color=(155,155,155), arr_width=10):
    
    split_perc = split(n_split, low_perc=low_perc)
    split_perc = [x * complete for x in split_perc]
    
    for index in range(n_split):
        degree_start = int(360 * split_perc[2*index]) + shift
        degree_end = int(360 * split_perc[2*index+1]) + shift

        cv2.ellipse(img, center, (radius, radius), 0, 
                    degree_start, degree_end, 
                    color=color, thickness=width)
    
    deg = int(360 * split_perc[0]) + shift
    point = (center[0] + int(radius * np.cos(deg * np.pi / 180)),
             center[1] + int(radius * np.sin(deg * np.pi / 180)))
    p1 = (point[0] + int(arr_len * np.cos((deg+45+90+7.5) * np.pi / 180)),
          point[1] + int(arr_len * np.sin((deg+45+90+7.5) * np.pi / 180)))
    p2 = (point[0] + int(arr_len * np.cos((deg-45+90+7.5) * np.pi / 180)),
          point[1] + int(arr_len * np.sin((deg-45+90+7.5) * np.pi / 180)))
    
    cv2.line(img, point, p1, arr_color, arr_width)
    cv2.line(img, point, p2, arr_color, arr_width)


def generate_love(size=512, point=(256,256), radius=50,
                  n_split=3, low_perc=0.6, 
                  color=(155,155,155), width=10,
                  arr_color=(155,155,155), arr_width=10):
    
    center = (point[0] - int(radius * 0.707),
              point[1] + int(radius * 0.707))
    generate_circle(size=size, center=center, radius=radius,
                  n_split=n_split, low_perc=low_perc, 
                  complete=0.48, shift=135,
                  color=color, width=width)

    center = (point[0] + int(radius * 0.707),
              point[1] + int(radius * 0.707))
    generate_circle(size=size, center=center, radius=radius,
                  n_split=n_split, low_perc=low_perc, 
                  complete=0.48, shift=225,
                  color=color, width=width)
    
    start = (point[0] - int(radius * 0.707 * 2),
             point[1] + int(radius * 0.707 * 2))
    end = (point[0], point[1] + int(radius * 0.707 * 4))
    generate_line(size=size, 
                  start=start, 
                  end=end,
                  n_split=n_split, low_perc=low_perc, 
                  color=arr_color, width=arr_width)
    
    start = (point[0] + int(radius * 0.707 * 2),
             point[1] + int(radius * 0.707 * 2))
    end = (point[0], point[1] + int(radius * 0.707 * 4))
    generate_line(size=size, 
                  start=start, 
                  end=end,
                  n_split=n_split, low_perc=low_perc, 
                  color=arr_color, width=arr_width)
    

def generate_coffee(size=512, point=(256,256), radius=100,
                    n_split=3, low_perc=0.6, 
                    color=(155,155,155), width=10):
    
    generate_line(size=size, 
                  start=point, 
                  end=(point[0]+2*radius, point[1]),
                  n_split=n_split, low_perc=low_perc, 
                  color=color, width=width)
    
    generate_line(size=size, 
                  start=point, 
                  end=(point[0], point[1]+radius),
                  n_split=n_split, low_perc=low_perc, 
                  color=color, width=width)
    
    generate_line(size=size, 
                  start=(point[0]+2*radius, point[1]),
                  end=(point[0]+2*radius, point[1]+radius),
                  n_split=n_split, low_perc=low_perc, 
                  color=color, width=width)
    
    bias = 0.3
    center = (point[0] + radius, 
              point[1] + int(radius - radius * (bias-0.05)))
    shift = np.arctan(bias) * 180 / np.pi
    complete = (180 - 2 * shift) / 360
    generate_circle(size=size, center=center, radius=int(radius*1.04),
                  n_split=n_split, low_perc=low_perc, 
                  complete=complete, shift=shift,
                  color=color, width=width)
    
    center = (point[0] + 2 * radius, 
              point[1] + int(radius * 0.5))
    generate_circle(size=size, center=center, radius=int(radius/2),
                  n_split=n_split, low_perc=low_perc, 
                  complete=0.5, shift=270,
                  color=color, width=width)
    
    start = (point[0] + int(radius * 0.5), 
             point[1] - int(radius * 0.2))
    end = (start[0] + int(radius * 0.7 * 0.5), 
           start[1] - int(radius * 0.7 * 0.866))
    cv2.line(img, start, end, color, width)
    
    start = (point[0] + int(radius * 1), 
             point[1] - int(radius * 0.2))
    end = (start[0] + int(radius * 0.3 * 0.5), 
           start[1] - int(radius * 0.3 * 0.866))
    cv2.line(img, start, end, color, width)
    
    start = (point[0] + int(radius * 1.5), 
             point[1] - int(radius * 0.2))
    end = (start[0] + int(radius * 0.4 * 0.5), 
           start[1] - int(radius * 0.4 * 0.866))
    cv2.line(img, start, end, color, width)





size = 1024
img = np.zeros((size, size, 3), np.uint8)

"""
generate_square(size=size, lefttop=(100,100), rightdown=(800,800),
              n_split=2, low_perc=0.7,
              color=(155,155,155), width=20)
"""
"""
generate_triangle(size=size, top=(100,512), length=800,
              n_split=2, low_perc=0.7,
              color=(155,155,155), width=20)
"""
"""
generate_circle(size=size, center=(512,512), radius=400,
                n_split=4, low_perc=0.8, complete=0.75, shift=45,
                color=(155,155,155), width=20)
"""
"""
generate_circ_with_array(size=size, center=(512,512), radius=400,
                n_split=4, low_perc=0.8, complete=0.8, shift=90,
                color=(155,155,155), width=20,
                arr_len = 100, arr_color=(155,155,155), arr_width=25)
"""
"""
generate_love(size=size, point=(512,192), radius=200,
              n_split=2, low_perc=0.8, 
              color=(155,155,155), width=20,
              arr_color=(155,155,155), arr_width=20)
"""

generate_coffee(size=size, point=(200,300), radius=300,
                n_split=1, low_perc=0.9, 
                color=(155,155,155), width=20)


img = cv2.GaussianBlur(img, (9,9), 0)
plt.imshow(255 - img, 'brg')
#cv2.imwrite('C:/Users/kongz/Desktop/DeeCamp/project/haha.png', img)