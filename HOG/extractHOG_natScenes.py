#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul
# @Date:   2015-10-02 13:08:32
# @Last Modified by:   Athul
# @Last Modified time: 2015-10-02 13:38:17

import numpy as np
import scipy.misc
from skimage import color, exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')

imgBaseDir = '../dataset/naturalScenes/'
featureBaseDir = 'features/naturalScenes'

imgDir = ['street/', 'forest/', 'Opencountry/', 'highway/', 'tallbuilding/', 'mountain/', 'inside_city/', 'coast/']

# Loop through each directory
# len(imgDir)
for cls in xrange(1):
    classDir = imgBaseDir + imgDir[cls]
    for imFile in os.listdir(classDir):
        img = scipy.misc.imread(classDir+imFile)
        img = color.rgb2gray(img)
        feature, hog_img = hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualise=True, normalise=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.axis('off')
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title("Input Image")

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 0.02))
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
plt.show()
