'''
author:jch
date: 2018/8/16
'''
import numpy as np
import pickle
import jieba
import warnings
import os
warnings.filterwarnings('error')

def getStopWords():
    f = open('stopwords.txt','r')
    lines = f.readlines()
    stopWords = []
    for line in lines:
        stopWords.append(line[0:-1])
    return stopWords

stopWords = getStopWords()

def removeStopWords(keyWords):
    global stopWords
    returnWords = []
    for keyWord in keyWords:
        if keyWord not in stopWords:
            returnWords.append(keyWord)
    return returnWords

def cos(v1, v2):
    return (np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2)))

#print(cos(vector[model['菜刀']],vector[model['婚礼']]))

def compute(inX, dataSet,norm=1):
    global vector, model, words
    dataSetSize = dataSet.shape[0]
    #计算距离
    result = np.zeros([dataSetSize])
    for i in range(dataSetSize):
##        if i %10000 ==0:
##            print(i)
        try:
            result[i] = cos(inX,dataSet[i])  ** norm
        except RuntimeWarning:
            result[i] = 0
            continue
    sortedDistIndicies = (-result).argsort(-1)
    return sortedDistIndicies,result

class word2vec:
    def __init__(self,targetWords, loadPath,norm = 1):
        self.targetWords = targetWords
        self.norm = norm
        f = open(loadPath, 'rb')
        self.model = pickle.load(f)
        self.vector = pickle.load(f)
        f.close()
        self.words = []
        for word in self.model:
            self.words.append(word)
        self.get_targetWords_vector()
        return
    def most_similar(self,word,topn=100):
        similar_words = []
        try:
            sort,result = compute(self.vector[self.model[word]],self.vector)
            for i in range(1,topn+1):
                similar_words.append((self.words[sort[i]],result[sort[i]]))
            return similar_words
        except KeyError:
            return False

    def get_feature_vector(self,topn = 256):
        feature_words = []
        feature_vector = np.zeros([topn * len(self.targetWords),300])
        for word in self.targetWords:
            for simi_word in self.most_similar(word,topn):
                feature_words.append(simi_word[0])
        print(len(feature_words),topn * len(self.targetWords))
        for i in range(topn * len(self.targetWords)):
            feature_vector[i] = self.vector[self.model[feature_words[i]]]
        self.feature_vector = feature_vector
        return feature_vector

    def get_targetWords_vector(self, topn=256):
        self.get_feature_vector(topn)
        print("feature_vector's shape:", self.feature_vector.shape)
        self.targetWords_vector = np.zeros([len(self.targetWords),len(self.feature_vector)])
        for i in range(len(self.targetWords)):
            self.targetWords_vector[i] = compute(self.vector[self.model[self.targetWords[i]]],self.feature_vector,self.norm)[1]
        return self.targetWords_vector

    def get_sentence_vector(self,keyWords):
        keyWords_length = len(keyWords)
        temp = np.zeros([keyWords_length,len(self.feature_vector)]) - 1
        num = 0
        for i in range(keyWords_length):
            try:
                temp[i] =  compute(self.vector[self.model[keyWords[i]]],self.feature_vector,self.norm)[1]
                num = num + 1
            except KeyError:
                continue
        if num ==0:
            return False
        sentence_vector = np.max(temp,axis=0)
        return sentence_vector


def regularization(inputs):
##    minimun = np.min(inputs)
##    maximun = np.max(inputs)
##    extent = abs(maximun - minimun) + 0.00000001
##    outputs = (inputs - minimun) / extent
    std = np.std(inputs)
    mean = np.mean(inputs)
    outputs = (inputs - mean) / std
    para = np.exp(5 * std)
    return outputs, para


def classify1(keyWords,model):
    sentence_vector = model.get_sentence_vector(keyWords)
    right_type = type(np.zeros([1]))
    if type(sentence_vector) != right_type:
        return False,False
    classify_num = len(model.targetWords)
    classify_result = np.zeros(classify_num)
    for i in range(classify_num):
        classify_result[i] = cos(sentence_vector,model.targetWords_vector[i])
    classify_result, para = regularization(classify_result)
    classify_result = np.exp(classify_result)
    return classify_result, para


# max
def classify2(keyWords,model):
    classify_num = len(model.targetWords)
    classify_result = np.zeros(classify_num)
    num = 0
    for i in range(len(model.targetWords)):
        max_cos = -1
        for keyWord in keyWords:
            try:
                max_cos = max(max_cos,cos(model.vector[model.model[keyWord]],model.vector[model.model[model.targetWords[i]]]))
                num = num + 1
            except KeyError:
                continue
        classify_result[i] = max_cos
    if num == 0:
        return False,False
    classify_result, para = regularization(classify_result)
    classify_result = np.exp(classify_result)
    return classify_result, para


## average
def classify3(keyWords,model):
    classify_num = len(model.targetWords)
    classify_result = np.zeros(classify_num)
    num = 0
    sum_cos = 0
    for i in range(len(model.targetWords)):
        sum_cos = 0
        num = 0
        for keyWord in keyWords:
            try:
                cos_vec = cos(model.vector[model.model[keyWord]],model.vector[model.model[model.targetWords[i]]])
                sum_cos += cos_vec * abs(cos_vec)
                num = num + 1
            except KeyError:
                continue
        average_cos = sum_cos / num
        classify_result[i] = average_cos
    if num == 0:
        return False,False
    classify_result, para = regularization(classify_result)
    classify_result = np.exp(classify_result)
    return classify_result, para

##alpha = 1.0
##beta = 0.5
##gama = 1.0

def sort_classify(classify):
    ## 降序
    ordered_classify = []
    order = (-classify).argsort(-1)
    for i in range(len(classify)):
        ordered_classify.append((targetWords[order[i]],classify[order[i]]))
##    for i in range(len(ordered_classify)):
##        print(ordered_classify[i][0],':',ordered_classify[i][1])
    return ordered_classify


def classify0(sentence,model):
    keyWords = jieba.lcut(sentence,cut_all = True)
    keyWords = removeStopWords(keyWords)

    classify_1, alpha = classify1(keyWords,model)
    if alpha == False:
        return False
    classify_2, beta = classify2(keyWords,model)
    if beta == False:
        return False
    classify_3, gama = classify3(keyWords,model)
    if gama == False:
        return False

##    sort_classify(classify_1)
##    print(alpha,'-'*30)
##    sort_classify(classify_2)
##    print(beta,'-'*30)
##    sort_classify(classify_3)
##    print(gama,'-'*30)
    
    classify_result = (classify_1 * alpha + classify_2 * beta + classify_3 * gama) / (alpha + beta + gama)
    classify_result = sort_classify(classify_result)
    return classify_result

# sgns.weibo.300d
# sgns.wiki.bigram.300d
loadPath = 'word2vec\sgns.weibo.300d'
#targetWords = ['飞机','蛋糕','青蛙','鼠标','刀','吉他','家猫','球','鲸鱼','口红']
targetWords = ['苹果', '公交车', '电脑', '玫瑰花', '猫' ,'绵羊' ,'烤鸭', '鸡']




if __name__ == "__main__":
    model = word2vec(targetWords, loadPath)
    while True:
        sentence = input("请输入描述:")
        classify_result = classify0(sentence,model)
        if type(classify_result) ==list:
            for i in range(len(classify_result)):
             print(classify_result[i][0],':',classify_result[i][1])
        elif classify_result == False:
            print("输入不合法")
