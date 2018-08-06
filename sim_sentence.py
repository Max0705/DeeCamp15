# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:18:21 2018

@author: kongz
"""
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import gensim

def cos(v1, v2):
    return np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))


model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

print('type model:', type(model))

"""
s = 'apple'
try:
    y2 = model.most_similar(s, topn=10)
    for item in y2:
        print(item[0], item[1])
    print("-----\n")
except:
    print("word '%s' not in vocabulary"%(s))
    print("-----\n")
 """

wordList = model.wv.index2word

label_list = ['banana', 'bridge', 'camera', 'hand', 'sheep',
              'computer', 'flower', 'knife', 'moon', 'rain']
label_list = [w for w in label_list if w in wordList]
label_vec = [[] for i in range(len(label_list))]

sentence = 'part of arm with five fingers'
sen_list = [w for w in sentence.split(' ') if w in wordList]
sen_vec = []

for wi, w in enumerate(wordList):
    if wi % 100000 == 0:
        print('word index:', wi)
    max_sim = max([model.similarity(w, s) for s in sen_list])
    sen_vec.append(max_sim)
    for i, label in enumerate(label_list):
        if False:
            label_vec[i].append(model.similarity(w, label))

print('-----------------------')

alpha, beta = 1, 0.5
result = [0 for label in label_vec]

for i, label in enumerate(label_list):
    
    cos_sim = cos(sen_vec, label_vec[i])
    result[i] += cos_sim
    
    max_sim = max([model.similarity(label, w) for w in sen_list])
    result[i] += alpha * max_sim
    
    avrg_sim = sum([model.similarity(label, w) for w in sen_list]) / len(sen_list)
    result[i] += beta * avrg_sim
    
    result[i] /= (1 + alpha + beta)
    print(label_list[i] + ':', 
          str(result[i])[:5], 
          str(cos_sim)[:5],
          str(max_sim)[:5], 
          str(avrg_sim)[:5])

