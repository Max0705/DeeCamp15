# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:14:43 2018

@author: kongz
"""

import sys
import numpy as np
import cv2
import imageio
import copy
import random
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


def update(last_img, new_img, label, labels,
           center, width, height, 
           nrow=1, extra=[0], 
           inverse=False):
    
    ans = last_img
    x1 = max(0, center[0] - int(height/2))
    x2 = min(last_img.shape[0], center[0] + int(height/2) + 1)
    y1 = max(0, center[1] - int(width/2))
    y2 = min(last_img.shape[1], center[1] + int(width/2) + 1)
    for i in range(x1, x2):
        for j in range(y1, y2):
            if labels[i][j] == label:
                    ans[i][j] = (new_img[i][j])[::-1]
        
    if nrow > 1:
        if nrow != len(extra):
            raise(TypeError)
        small_height = int(height / nrow)
        
        for index in range(nrow):
            
            x_new1 = x1 + index * small_height
            if index == nrow - 1:
                x_new2 = x2
            else:
                x_new2 = min(last_img.shape[0], x_new1 + small_height + 1)
            
            if inverse:
                y_new2 = max(0, y2 - width)
                y_new1 = max(0, y_new2 - extra[index] - 1)
            else:
                y_new1 = y2 + 0
                y_new2 = min(last_img.shape[1], y_new1 + extra[index] + 1)
            
            if y_new2 > y_new1:
                for i in range(x_new1, x_new2):
                    for j in range(y_new1, y_new2):
                        if labels[i][j] == label:
                            ans[i][j] = (new_img[i][j])[::-1]
    return ans
   
    
def delete_gray(img):
    interval = range(237, 243)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color = img[i][j]
            if color[0] in interval and color[1] in interval and color[2] in interval:
                img[i][j] = [255, 255, 255]
    print('Gray deleted')


def add_circ(img, center=(256,256), radius=100, color=(155,155,155), width=5):
    cv2.circle(img, center, radius, color, width)
    print('Circle generated')
        

def generate_gif(sketch, fig, outfile,
                 n_clusters=1, labels=None,
                 n_batch=8, freq=50, mode='horizontal', nrow=20):
    h, w, ch = sketch.shape
    if sketch.shape != fig.shape:
        raise(ValueError)
    
    if labels is None:
        n_clusters = 1
        labels = np.zeros([h, w])
    
    img = sketch
    height = int(h / n_batch)
    width = int(w / freq)
    with imageio.get_writer(outfile, mode='I') as writer:
        for label in range(n_clusters):
            print('\n----------label:', label, '----------')
            for batch in range(n_batch):
                print('batch:', batch)
                for index in range(freq):
                    if mode == 'horizontal':
                        
                        inverse = batch % 2 == 1
                        if inverse:
                            x = int(height * (batch + 0.5))
                            y = int(width * (freq - index - 0.5))
                        else:
                            x = int(height * (batch + 0.5))
                            y = int(width * (index + 0.5))
                        
                        extra = [0 + int(40*random.random()) for i in range(nrow)]
                        img = update(img, fig, label, labels,
                                     [x,y], width, height, 
                                     nrow=nrow, extra=extra,
                                     inverse=inverse)
                        
                        # img_add = copy.deepcopy(img)
                        #add_circ(img_add, (y,x), radius=int(height/3))
                        writer.append_data(img)
                        
    print('GIF generated')


def simplify(img, method='KMeans', n_clusters=5, change_style=True):
    h, w, ch = img.shape
    reshaped = copy.deepcopy(img).reshape(h * w, ch)
    reshaped = reshaped / 255.0
    
    if method == 'KMeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(reshaped)
        labels = kmeans.labels_.reshape(h, w)
        centers = kmeans.cluster_centers_
    elif methods == 'DBScan':
        db = DBSCAN(eps=0.3, min_samples=5).fit(reshaped)
        labels = db.labels_.reshape(h, w)
        # centers = ?
    centers = centers * 255.0
    if change_style:
        for i in range(h):
            for j in range(w):
                img[i][j] = centers[labels[i][j]]
    
    return n_clusters, labels, centers
    

    
    
sketch = cv2.imread('./gif/input/car_gen_sketch.png')
fig = cv2.imread('./gif/input/car_gen.png')
outfile = './gif/output/car_gen_gif.gif'
#delete_gray(fig)

n_clusters, labels = 1, None
n_clusters, labels, centers = simplify(fig, method='KMeans', n_clusters=5, 
                                       change_style=True)

generate_gif(sketch, fig, outfile,
             n_clusters, labels,
             n_batch=4, freq=32, mode='horizontal', nrow=8)


#cv2.imwrite('./gif/output/test.jpg', fig)
