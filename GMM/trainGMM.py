#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul
# @Date:   2015-10-29 12:50:11
# @Last Modified by:   Athul
# @Last Modified time: 2015-10-29 13:25:33
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import gzip
import os
from sklearn import mixture
import sklearn.metrics as metrics

dataLoc = '../HOG/features/naturalScenes_HOG_features.pkl.gz'
modelFile = 'results/gmm_10_natscenes.pkl'

with gzip.open(dataLoc, 'rb') as f:
    trainData, valData, testData = pickle.load(f)
    f.close()

trainFeatures, trainLabels = trainData
valFeatures, valLabels = valData
testFeatures, testLabels = testData

trainFeatures = np.vstack((trainFeatures, valFeatures))
trainLabels = np.append(trainLabels, valLabels)

del trainData, testData, valData, valFeatures, valLabels

# ====================== train GMM ===========================

n_gaussians = np.array([10, 10, 10, 10, 10, 10, 10, 10])

classes = np.unique(trainLabels)
models = {}
for cls in classes:
    models[cls] = mixture.GMM(n_components=n_gaussians[cls])
    models[cls].fit(trainFeatures[trainLabels==cls])

with open(modelFile, 'w') as f:
    pickle.dump(models, f)

print('Training of GMM done and dumped the model to file '+modelFile)


