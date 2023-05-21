from __future__ import unicode_literals
from hazm import *
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import pyplot as plt
from operator import itemgetter
import math
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm
from hazm.utils import stopwords_list

additiveSmoothing = True

TECHNOLOGY = 0
SPORT = 1
ACCIDENTS = 2
HEALTH = 3
POLITICIAL = 4
ART_CULTURE = 5

DataFrame = pd.read_csv('train.csv')
nor = Normalizer()
wordT = WordTokenizer()
lemma = Lemmatizer()
stopWords = stopwords_list()
WritingWords = [",",".","/","?",":","=","+","-","_",")","(","&","!","،","؛","«","»"]
dictTemp = {}
numberOfWords = []
probabilities = []
numOfTrainNews = {'فناوری':0, 'ورزشی':0, 'حوادث':0, 'سلامت':0, 'سیاسی':0, 'فرهنگی/هنری':0}
numOfTestNews = {'فناوری':0, 'ورزشی':0, 'حوادث':0, 'سلامت':0, 'سیاسی':0, 'فرهنگی/هنری':0}
labelMap = ['فناوری' ,'ورزشی' ,'حوادث' ,'سلامت' ,'سیاسی' ,'فرهنگی/هنری']
numOfAllDetected = []
correctDetectedClass = []

for i in range(6):
    correctDetectedClass.append(0)
for i in range(6):
    probabilities.append(dictTemp.copy())
for i in range(6):
    numberOfWords.append(0)
for i in range(6):
    numOfAllDetected.append(0)

def applyNumOfWord(_label_, index, word):
    if (word in probabilities[index]):
        probabilities[index][word] += 1
        numberOfWords[index] += 1
    else:
        probabilities[index][word] = 1
        numberOfWords[index] += 1

def calculateAccMacMicWei(F1_0, F1_1, F1_2, F1_3, F1_4, F1_5):
    Accuracy = (correctDetectedClass[TECHNOLOGY] + correctDetectedClass[SPORT] +
                correctDetectedClass[ACCIDENTS] + correctDetectedClass[HEALTH] + correctDetectedClass[POLITICIAL] + 
                correctDetectedClass[ART_CULTURE]) / (numOfTestNews[labelMap[TECHNOLOGY]] + numOfTestNews[labelMap[SPORT]] +
                numOfTestNews[labelMap[ACCIDENTS]] + numOfTestNews[labelMap[HEALTH]] + 
                numOfTestNews[labelMap[POLITICIAL]] + numOfTestNews[labelMap[ART_CULTURE]])

    Macro = (F1_0 + F1_1 + F1_2 + F1_3 + F1_4 + F1_5) / 6
    numOfAllTestNews = numOfTestNews['حوادث'] + numOfTestNews['سلامت'] + numOfTestNews['سیاسی'] + numOfTestNews['فرهنگی/هنری'] + numOfTestNews['فناوری'] + numOfTestNews['ورزشی']
    Weighted = (F1_2*(numOfTestNews['حوادث'] / numOfAllTestNews)) + (F1_3*(numOfTestNews['سلامت'] / numOfAllTestNews)) + (F1_4*(numOfTestNews['سیاسی'] / numOfAllTestNews)) + (F1_5*(numOfTestNews['فرهنگی/هنری'] / numOfAllTestNews)) + (F1_0*(numOfTestNews['فناوری'] / numOfAllTestNews)) + (F1_0*(numOfTestNews['ورزشی'] / numOfAllTestNews))
    print("Accuracy : ", Accuracy)
    print("Micro : ", Accuracy)
    print("Macro : ", Macro)
    print("Weighted : ", Weighted)

def calculatePrecReclF1(_label_):
    precision = correctDetectedClass[_label_] / numOfAllDetected[_label_]
    recall = correctDetectedClass[_label_] / numOfTestNews[labelMap[_label_]]
    F1 = 2 * ((precision * recall) / (precision + recall))
    print("-------", labelMap[_label_], "-------")
    print("precision : ", precision)
    print("recall : ", recall)
    print("F1 : ", F1)
    return F1

def validityCheck():
    count = 0
    for p in probabilities:
        chart = dict(sorted(p.items(), key=itemgetter(1), reverse=True)[:5])
        keys = chart.keys()
        values = chart.values()
        plt.bar(keys, values)
        plt.xlabel(labelMap[count])
        plt.ylabel('تعداد')
        plt.show()
        count += 1

for index, j in DataFrame.iterrows():
    news = j['content']
    label = j['label']
    news = nor.normalize(news)
    news = wordT.tokenize(news)
    numOfTrainNews[label] += 1
    k = 0
    while (k != len(news)):
        count = 0
        if (news[k] in stopWords or news[k] in WritingWords):
            news.remove(news[k])
            count = 1
        if (count != 1):
            news[k] = lemma.lemmatize(news[k])
            k += 1
    for i in news:
        if (label == 'فناوری'):
            applyNumOfWord(label, TECHNOLOGY, i)
        elif (label == 'ورزشی'):
            applyNumOfWord(label, SPORT, i)
        elif (label == 'حوادث'):
            applyNumOfWord(label, ACCIDENTS, i)
        elif (label == 'سلامت'):
            applyNumOfWord(label, HEALTH, i)
        elif (label == 'سیاسی'):
            applyNumOfWord(label, POLITICIAL, i)
        elif (label == 'فرهنگی/هنری'):
            applyNumOfWord(label, ART_CULTURE, i)

DataFrameTest = pd.read_csv('test.csv')

for index, j in DataFrameTest.iterrows():
    news = j['content']
    label = j['label']
    labelNum = 0
    numOfTestNews[label] += 1
    if (label == 'فناوری'):
        labelNum = TECHNOLOGY
    elif (label == 'ورزشی'):
        labelNum = SPORT
    elif (label == 'حوادث'):
        labelNum = ACCIDENTS
    elif (label == 'سلامت'):
        labelNum = HEALTH
    elif (label == 'سیاسی'):
        labelNum = POLITICIAL
    elif (label == 'فرهنگی/هنری'):
        labelNum = ART_CULTURE
    news = nor.normalize(news)
    news = wordT.tokenize(news)
    k = 0
    while (k != len(news)):
        count = 0
        if (news[k] in stopWords or news[k] in WritingWords):
            news.remove(news[k])
            count = 1
        if (count != 1):
            news[k] = lemma.lemmatize(news[k])
            k += 1
    posteriorProbabilities = []
    for i in range(6):
        posteriorProbabilities.append(0)
    for c in range(6):
        likelihood = 0
        for w in news:
            if (w in probabilities[c]):
                likelihood = likelihood + math.log(probabilities[c][w] / numberOfWords[c])
            elif (additiveSmoothing == True):
                likelihood = likelihood + math.log(1 / (numberOfWords[c] + len(probabilities[c])))
        posteriorProbabilities[c] = likelihood + math.log(1/6)
    probableClass = posteriorProbabilities.index(max(posteriorProbabilities))
    numOfAllDetected[probableClass] += 1
    if (labelNum == probableClass):
        correctDetectedClass[probableClass] += 1

F1_0 = calculatePrecReclF1(TECHNOLOGY)
F1_1 = calculatePrecReclF1(SPORT)
F1_2 = calculatePrecReclF1(ACCIDENTS)
F1_3 = calculatePrecReclF1(HEALTH)
F1_4 = calculatePrecReclF1(POLITICIAL)
F1_5 = calculatePrecReclF1(ART_CULTURE)

calculateAccMacMicWei(F1_0, F1_1, F1_2, F1_3, F1_4, F1_5)
validityCheck()