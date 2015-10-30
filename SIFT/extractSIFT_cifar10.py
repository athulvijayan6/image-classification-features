#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul Vijayan
# @Date:   2015-08-05 19:59:29
# @Last Modified by:   Athul
# @Last Modified time: 2015-10-30 11:48:22
import numpy as np
import Image
import os
from scipy.cluster.vq import whiten, kmeans, vq
from random import shuffle
import gzip
try:
    import cPickle as pickle
except:
    import pickle

clsNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']
imgBaseDir = '../dataset/CIFAR-10/'

dataFile = 'features/cifar10_gray_SIFT_features.pkl.gz'

os.system('rm -r .tmp; mkdir .tmp')

trainData = testData = []
for batchFile in os.listdir(imgBaseDir):
    if batchFile.find('_batch') != -1:
        with open(imgBaseDir+batchFile, 'rb') as b:
            d = pickle.load(b)
            b.close()
        pixels = d['data']
        labels = d['labels']
        
        for idx in xrange(10000):
            img = pixels[idx] 
            img = np.reshape(img, (3, 32, 32))
            img = np.transpose(img, [1, 2, 0])
            img = Image.fromarray(img)
            img = img.convert('L')
            img.save('.tmp/tempImg.pgm')
            # Execute the SIFT extractor
            os.system('./siftLowe/sift <.tmp/tempImg.pgm >.tmp/keys.key')
            # Read the output files
            with open ('.tmp/keys.key', 'rb') as f:
                numDesc, lenDesc = [int(i) for i in f.readline().split()]
                locs = np.zeros((4,))
                desc = lf = np.zeros((lenDesc,), dtype=np.float)
                i = lenDesc
                j = 0
                for line in f.readlines():
                    vals = [float(n) for n in line.strip().split()]
                    if i == lenDesc:
                        locs = np.vstack((locs, vals))
                        lf = lf/np.sqrt(np.sum(lf**2))
                        desc = np.vstack((desc, lf))
                        i = 0
                    else:
                        lf[i:i+len(vals)] = vals
                        i += len(vals)
                f.close()
            lf = lf/np.sqrt(np.sum(lf**2))
            desc = np.vstack((desc, lf))
            locs = locs[1:]
            desc = desc[2:]
            if batchFile[:-2] == 'data_batch':
                trainData.append((locs, desc, labels[idx]))
            elif batchFile == 'test_batch':
                testData.append((locs, desc, labels[idx]))
                
            os.system('rm .tmp/tempImg.pgm .tmp/keys.key')

for desc in trainData:
    try:
        trainFeatures = np.vstack((trainFeatures, desc[1]))
    except:
        trainFeatures = desc[1]

# Collect all training + validation data to form a dictionary using k-means
dictionary = kmeans(whiten(trainFeatures), featureDim)[0]
del trainFeatures

featureDim = 512

# Now form a histogram for each images as final feature vector
for img in trainData:
    imData = img[1]
    feature = np.zeros((featureDim, ))
    for i in xrange(imData.shape[0]):
        c = np.reshape(imData[i], (1, imData[i].size))
        id = vq(c, dictionary)[0][0]
        feature[id] += 1
    try:
        trainFeatures = np.vstack((trainFeatures, feature))
        trainLabels = np.append(trainLabels, img[2])
    except:
        trainFeatures = feature
        trainLabels = np.array([img[2]])


for img in testData:
    imData = img[1]
    feature = np.zeros((featureDim, ))
    for i in xrange(imData.shape[0]):
        c = np.reshape(imData[i], (1, imData[i].size))
        id = vq(c, dictionary)[0][0]
        feature[id] += 1
    try:
        testFeatures = np.vstack((testFeatures, feature))
        testLabels = np.append(testLabels, img[2])
    except:
        testFeatures = feature
        testLabels = np.array([img[2]])

valFeatures = trainFeatures[ :int(0.1*trainFeatures.shape[0])]
valLabels = trainLabels[ :int(0.1*trainLabels.shape[0])]

trainFeatures = trainFeatures[int(0.1*trainFeatures.shape[0]): ]
trainLabels = trainLabels[int(0.1*trainLabels.shape[0]): ]

with gzip.open(featureFile, 'wb') as f:
    pickle.dump(((trainFeatures, trainLabels), (valFeatures, valLabels), (testFeatures, testLabels)), f)
    f.close()
print("Dumped features into "+dataFile)