#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul
# @Date:   2015-10-29 13:20:58
# @Last Modified by:   Athul
# @Last Modified time: 2015-10-29 13:26:30
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

testFeatures, testLabels = testData
del trainData, testData, valData

with open(modelFile, 'rb') as f:
    models = pickle.load(f)
classes = np.array(models.keys())

# ====================== test GMM ====================================
scores = np.zeros((testFeatures.shape[0], classes.size))

for cls in classes:
    scores[:, cls] = models[cls].score(testFeatures)

predLabels = np.argmax(scores, axis = 1)
print('Done testing for the input')
# ================================ Performance metrices ==============
accuracy = metrics.accuracy_score(testLabels, predLabels)
print('Accuracy: '+str(accuracy*100))
cm = metrics.confusion_matrix(testLabels, predLabels)
print cm
