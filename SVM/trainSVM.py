#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul
# @Date:   2015-10-05 16:01:10
# @Last Modified by:   Athul
# @Last Modified time: 2015-10-29 13:10:31

import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import gzip
import os

dataLoc = '../HOG/features/naturalScenes_HOG_features.pkl.gz'

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

# ============================= Train SVM ========================
print('scaling data')
command = './libsvm/svm-scale -l -1 -u 1 -s results/range {0} > {1}.scale'
command = command.format(trainFile, trainFile)
os.system(command)

trainFile += '.scale'



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