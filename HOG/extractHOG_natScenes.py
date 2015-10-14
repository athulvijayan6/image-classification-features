#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul
# @Date:   2015-10-02 13:08:32
# @Last Modified by:   Athul Vijayan
# @Last Modified time: 2015-10-05 23:24:49

import numpy as np
import scipy.misc
from skimage import color, exposure
from skimage.feature import hog
import os
import gzip
try:
    import cPickle as pickle
except:
    import pickle

imgBaseDir = '../dataset/naturalScenes/'
featureBaseDir = 'features/naturalScenes'

imgDir = ['street/', 'forest/', 'Opencountry/', 'highway/', 'tallbuilding/', 'mountain/', 'inside_city/', 'coast/']
dataFile = 'features/naturalScenes_HOG_features.pkl.gz'

# Loop through each directory
for cls in xrange(len(imgDir)):
    i = 0
    classDir = imgBaseDir + imgDir[cls]
    filenames = os.listdir(classDir)
    for imFile in filenames:
        img = scipy.misc.imread(classDir+imFile)
        img = color.rgb2gray(img)
        feature = hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualise=False, normalise=True)
        feature = np.append(feature, cls)
        try:
            features = np.vstack((features, feature))
        except NameError:
            features = feature

np.random.shuffle(features)
total = features.shape[0]
trainFeatures = features[:0.7*total, :-1]
trainLabels = features[:0.7*total, -1]
valFeatures = features[0.7*total:0.8*total, :-1]
valLabels = features[0.7*total:0.8*total, -1]
testFeatures = features[0.8*total:, :-1]
testLabels = features[0.8*total:, -1]
with gzip.open(dataFile, 'wb') as f:
    pickle.dump(((trainFeatures, trainLabels), (valFeatures, valLabels), (testFeatures, testLabels)), f)
print("Dumped HOG features into "+dataFile)


