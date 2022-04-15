#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from sklearn.decomposition import FastICA, PCA
from scipy.spatial import distance

def readCenter(name):
    #read file
    rate1, data = wavfile.read(name)
    #center data
    data = data-np.mean(data)
    data = data/32768
    return data

def applyICA(data1, data2, _whiten):
    # Creating a matrix out of the signals
    X = np.c_[data1, data2]
    ica= FastICA(n_components=2, whiten = _whiten)
    #s is the transformed data 
    S_ = ica.fit_transform(X)
    return S_

def getChannels(data):
    r1 = []
    r2 = []
    for i in range(len(data)):
        r1.append(data[i][0])
        r2.append(data[i][0])
    return r1, r2

def printDistance(x, y):
    d = round(distance.cityblock(x, y), 2)
    return ("distance" + str(x) + str(y) + str(d)
    

data1 = readCenter("sounds_mixedX.wav")
data2 = readCenter("sounds_mixedY.wav")
S_ = applyICA(data1, data2, True)    
r1, r2 = getChannels(S_)
print(printDistance(data1, r1))
    



#printDistance(data2, r1)
#printDistance(data1, r2)
#printDistance(data2, r2)

#plt.plot(S_)
#plt.show()
#print(S_)
#plt.plot(realX)
#plt.plot(realY)
#actual = np.c_[realX, realY]
#f = open(filename, 'w')
#f.write(dist)
#with open(filename) as f:
    #f.write(dist)
    
#Record DIST(s1,r1), DIST(s1,r1), DIST(s2,r1), DIST(s2,r2) using the default "whiten = True" in FastICA