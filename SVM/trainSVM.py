#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul
# @Date:   2015-10-05 16:01:10
# @Last Modified by:   Athul
# @Last Modified time: 2015-10-06 09:46:09

import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import gzip
import os

dataLoc = '../SIFT/features/naturalScenes_SIFT_features.pkl.gz'

with gzip.open(dataLoc, 'rb') as f:
    trainData, valData, testData = pickle.load(f)
    f.close()

trainFeatures, trainLabels = trainData
valFeatures, valLabels = valData
testFeatures, testLabels = testData

trainFeatures = np.vstack((trainFeatures, valFeatures))
trainLabels = np.append(trainLabels, valLabels)

del trainData, testData, valData, valFeatures, valLabels

trainFile = 'results/train_svm_data'
modelFile = 'results/model_svm_01.model'
testFile = 'results/test_svm_data'
resultFile = 'results/results'

with open(trainFile, 'w') as f:
    for i in xrange(trainFeatures.shape[0]):
        line = str(trainLabels[i])
        for j in xrange(trainFeatures.shape[1]):
            l = ' {0}:{1}'
            l = l.format(j+1, trainFeatures[i, j])
            line += l
        line += '\n'
        f.write(line)
    f.close()

with open(testFile, 'w') as f:
    for i in xrange(testFeatures.shape[0]):
        line = str(testLabels[i])
        for j in xrange(testFeatures.shape[1]):
            l = ' {0}:{1}'
            l = l.format(j+1, testFeatures[i, j])
            line += l
        line += '\n'
        f.write(line)
    f.close()

# ============================= Train SVM ========================
print('scaling data')
command = './libsvm/svm-scale -l -1 -u 1 -s results/range {0} > {1}.scale'
command = command.format(trainFile, trainFile)
os.system(command)

command = './libsvm/svm-scale -r results/range {0} > {1}.scale'
command = command.format(testFile, testFile)
os.system(command)

trainFile += '.scale'
testFile += '.scale'


print('starting svm training')
s = 0
t = 2
d = 2
g = 0.0625
r = 0
c = 2
b = 0
v = 5
command = './libsvm/svm-train {8} {9} -s {0} -t {1} -d {2} -g {3} -r {4} -c {5} -b {6} -v {7}'
command = command.format(s, t, d, g, r, c, b, v, trainFile, modelFile)
os.system(command)

# # ============================== Test svm model===============
print('starting svm testing')
b = 0
command = './libsvm/svm-predict -b {0} {1} {2} {3}'
command = command.format(b, testFile, modelFile, resultFile)
os.system(command)