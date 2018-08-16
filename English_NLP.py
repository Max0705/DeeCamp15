# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:27:54 2018

@author: kongz
"""
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import gensim
import os


def cos(v1, v2):
    return np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))


def load_GoogleWord2Vec_model(path=''):
    try:
        model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(path, 'GoogleNews-vectors-negative300.bin'), 
                                                                binary=True)
        print('Model loading complete')
        print('type model:', type(model))
        return model
    except:
        print('Cannot load word2vec model')
        raise(ValueError)


def load_label_Sen2Vec(label_list):
    label_list = [w for w in label_list if w in model]
    label_vec = [[] for i in range(len(label_list))]
    
    wordList = []
    for label in label_list:
        sim_topn = model.most_similar(label, topn=1000)
        wordList += [item[0] for item in sim_topn]
    
    print('WordList complete')
    
    for wi, w in enumerate(wordList):
        if wi % 2000 == 0:
            print('word index:', wi)
        for i, label in enumerate(label_list):
            label_vec[i].append(model.similarity(w, label))
            
    print('Label Sen2Vec complete')
    
    return wordList, label_vec


def predict_label_of_sentence(label_list, sentence):
    sentence = sentence.lower()
    sen_list = [w for w in sentence.split(' ') if w in model]
    sen_vec = []
    for wi, w in enumerate(wordList):
        if wi % 2000 == 0:
            print('word index:', wi)
        max_sim = max([model.similarity(w, s) for s in sen_list])
        sen_vec.append(max_sim)
    
    print('Sentence Sen2Vec complete')
    
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
    
    return label_list[result.index(max(result))]


model_path = ''
model = load_GoogleWord2Vec_model(model_path)
label_list = ['banana', 'bridge', 'camera', 'hand', 'sheep',
              'computer', 'flower', 'knife', 'moon', 'rain']
wordList, label_vec = load_label_Sen2Vec(label_list)

sentence = 'metal tool to cut in the kitchen'
result = predict_label_of_sentence(label_list=label_list, sentence=sentence)
print('\nThe predicted label of sentence', '<', sentence, '>', 'is', '<', result, '>.')