#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul
# @Date:   2015-10-05 16:01:34
# @Last Modified by:   Athul
# @Last Modified time: 2015-10-29 13:11:08
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import gzip
import os
import sklearn.metrics as metrics

dataLoc = '../HOG/features/naturalScenes_HOG_features.pkl.gz'
with gzip.open(dataLoc, 'rb') as f:
    trainData, valData, testData = pickle.load(f)
    f.close()
testFeatures, testLabels = testData

del trainData, testData, valData

testFile = 'results/test_svm_data'
modelFile = 'results/model_svm_01.model'
resultFile = 'results/results'

with open(testFile, 'w') as f:
    for i in xrange(testFeatures.shape[0]):
        line = str(testLabels[i])
        for j in xrange(testFeatures.shape[1]):
            l = ' {0}:{1}'
            l = l.format(j+1, testFeatures[i, j])
            line += l
        line += '\n'
        f.write(line)
    trueLabels = np.array(trueLabels)
    f.close()

# # ============================== Test svm model===============
command = './libsvm/svm-scale -r results/range {0} > {1}.scale'
command = command.format(testFile, testFile)
os.system(command)

testFile += '.scale'

print('starting svm testing')

b = 0
command = './libsvm/svm-predict -b {0} {1} {2} {3}'
command = command.format(b, testFile, modelFile, resultFile)
os.system(command)

# =============================== Read results file =================
with open(resultFile, 'rb') as f:
    predLabels = [int(line.rstrip('\n')) for line in f]
    predLabels = np.array(predLabels)
    f.close()

# ================================ Performance metrices ==============
cm = metrics.confusion_matrix(testLabels, predLabels)
print cm

