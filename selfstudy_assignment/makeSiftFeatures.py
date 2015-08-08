#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul Vijayan
# @Date:   2015-08-05 19:59:29
# @Last Modified by:   Athul Vijayan
# @Last Modified time: 2015-08-05 20:33:03

import numpy as np
from scipy import misc
import os

imgDir = 'dataset/naturalScenes/'
featureDir = 'features/naturalScenes'

# Loop through each directory
for subdir, dirs, files in os.walk(imgDir):
    # Loop through each file
    for f in files:
        # f is the file
        img = misc.imread(f)
        print os.path.join(os.path.basename(subdir), f)