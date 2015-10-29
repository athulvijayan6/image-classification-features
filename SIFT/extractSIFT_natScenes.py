#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul Vijayan
# @Date:   2015-08-05 19:59:29
# @Last Modified by:   Athul
# @Last Modified time: 2015-10-29 12:01:54

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

imgBaseDir = '../dataset/naturalScenes/'
featureBaseDir = 'features/naturalScenes'

imgDir = ['street/', 'forest/', 'Opencountry/', 'highway/', 'tallbuilding/', 'mountain/', 'inside_city/', 'coast/']

featureFile = 'features/naturalScenes_SIFT_features.pkl.gz'

os.system('rm -r .tmp; mkdir .tmp')

# Loop through each directory
descs = []
for cls in xrange(len(imgDir)):
    classDir = imgBaseDir + imgDir[cls]
    filenames = os.listdir(classDir)
    for imFile in filenames:
        img = Image.open(classDir+imFile)
        img = img.resize((128, 128), Image.ANTIALIAS)
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
        descs.append((locs, desc, cls))
        os.system('rm .tmp/tempImg.pgm .tmp/keys.key')

shuffle(descs)
# Now we that have all data, use bag of words kind of feature representation
featureDim = 512
trainData = descs[:int(0.7*len(descs))]
valData = descs[int(0.7*len(descs)):int(0.8*len(descs))]
testData = descs[int(0.8*len(descs)):]

for desc in trainData:
    try:
        trainFeatures = np.vstack((trainFeatures, desc[1]))
    except:
        trainFeatures = desc[1]

for desc in valData:
    try:
        valFeatures = np.vstack((valFeatures, desc[1]))
    except:
        valFeatures = desc[1]

# Collect all training + validation data to form a dictionary using k-means
dictionary = kmeans(whiten(np.vstack((trainFeatures, valFeatures))), featureDim)[0]
del trainFeatures, valFeatures

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

for img in valData:
    imData = img[1]
    feature = np.zeros((featureDim, ))
    for i in xrange(imData.shape[0]):
        c = np.reshape(imData[i], (1, imData[i].size))
        id = vq(c, dictionary)[0][0]
        feature[id] += 1
    try:
        valFeatures = np.vstack((valFeatures, feature))
        valLabels = np.append(valLabels, img[2])
    except:
        valFeatures = feature
        valLabels = np.array([img[2]])

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

with gzip.open(featureFile, 'wb') as f:
    pickle.dump(((trainFeatures, trainLabels), (valFeatures, valLabels), (testFeatures, testLabels)), f)
    f.close()
print("Dumped features into "+featureFile)