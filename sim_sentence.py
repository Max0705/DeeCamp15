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

# wordList = model.wv.index2word

label_list = ['banana', 'bridge', 'camera', 'hand', 'sheep',
              'computer', 'flower', 'knife', 'moon', 'rain']
#label_list = ['airplane', 'taxi', 'car', 'truck', 'ship',
#              'boat', 'bike', 'walk', 'bus', 'sportscar']
label_list = [w for w in label_list if w in model]
label_vec = [[] for i in range(len(label_list))]

wordList = []
for label in label_list:
    sim_topn = model.most_similar(label, topn=1000)
    wordList += [item[0] for item in sim_topn]

print('wordList complete')

sentence = 'metal tool to cut in the kitchen'
sen_list = [w for w in sentence.split(' ') if w in model]
sen_vec = []

for wi, w in enumerate(wordList):
    if wi % 2000 == 0:
        print('word index:', wi)
    max_sim = max([model.similarity(w, s) for s in sen_list])
    sen_vec.append(max_sim)
    for i, label in enumerate(label_list):
        if True:
            label_vec[i].append(model.similarity(w, label))

print('sen2vec complete')
print('-----------------------')

alpha, beta = 1, 0.5
result = [0 for label in label_vec]

for i, label in enumerate(label_list):
    cos_sim = cos(sen_vec, label_vec[i])
    sim_vec = [model.similarity(label, w) for w in sen_list]
    max_sim = max(sim_vec)
    avrg_sim = np.sqrt(sum([x * abs(x) for x in sim_vec]) / len(sen_list))
    result[i] = (cos_sim + alpha * max_sim + beta * avrg_sim) / (1 + alpha + beta)
    print(label_list[i] + ':\t' + ('\t' if len(label_list[i]) < 7 else ''), 
          str(result[i])[:5], '\t',
          str(cos_sim)[:5],
          str(max_sim)[:5], 
          str(avrg_sim)[:5])

