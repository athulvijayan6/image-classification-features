#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul
# @Date:   2015-10-02 13:08:32
# @Last Modified by:   Athul
# @Last Modified time: 2015-10-29 17:08:45

import numpy as np
import scipy.misc
from skimage import color, exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import os
import gzip
try:
    import cPickle as pickle
except:
    import pickle

clsNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']
imgBaseDir = '../dataset/CIFAR-10/'
featureBaseDir = 'features/CIFAR-10'

dataFile = 'features/cifar10_gray_HOG_features.pkl.gz'

for f in os.listdir(imgBaseDir):
    if f.find('batch') != -1:
        with open(imgBaseDir+f, 'rb') as b:
            d = pickle.load(b)
            b.close()

        pixels = d['data']
        labels = d['labels']

        for idx in xrange(10000):
            img = pixels[idx] 
            img = np.reshape(img, (3, 32, 32))
            img = np.transpose(img, [1, 2, 0])
            img = color.rgb2gray(img)
            feature = hog(img, orientations=8, pixels_per_cell=(1, 1), cells_per_block=(1, 1), visualise=False, normalise=True)
            feature = np.append(feature, cls)
            if f[:-2] == 'data_batch':
                try:
                    trainFeatures = np.vstack((trainFeatures, feature))
                    np.append(trainLabels, labels[idx])
                except NameError:
                    trainFeatures = feature
                    trainLabels = np.array([labels[idx]])
            elif f == 'test_batch':
                try:
                    testFeatures = np.vstack((testFeatures, feature))
                    np.append(testLabels, labels[idx])
                except NameError:
                    testFeatures = feature
                    testLabels = np.array([labels[idx]])

valFeatures = trainFeatures[ :int(0.1*trainFeatures.shape[0])]
valLabels = trainLabels[ :int(0.1*trainLabels.shape[0])]

trainFeatures = trainFeatures[int(0.1*trainFeatures.shape[0]): ]
trainLabels = trainLabels[int(0.1*trainLabels.shape[0]): ]

with gzip.open(dataFile, 'wb') as f:
    pickle.dump(((trainFeatures, trainLabels), (valFeatures, valLabels), (testFeatures, testLabels)), f)
print("Dumped HOG features into "+dataFile)