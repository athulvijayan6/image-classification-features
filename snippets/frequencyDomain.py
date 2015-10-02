#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Athul Vijayan
# @Date:   2015-08-06 12:37:58
# @Last Modified by:   Athul Vijayan
# @Last Modified time: 2015-08-06 19:40:11

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

img = misc.imread('../test01.jpg', flatten=True)

fimg = np.fft.fft2(img)

lowPass = np.ones(fimg.shape)
indices = np.abs(fimg) > np.amax(np.abs(fimg))
lowPass[indices] = 0
lowPassFft = fimg*lowPass

highPass = np.ones(fimg.shape)
indices = np.abs(fimg) < np.amin(np.abs(fimg))/0.001
highPass[indices] = 0
highPassFft = fimg*highPass

lowPassImg = np.fft.ifft2(lowPassFft)
highPassImg = np.fft.ifft2(highPassFft)

fig , plots = plt.subplots(2, 3)
plots[0, 0].imshow(img, cmap='gray')
plots[0, 0].set_title('Original image')
plots[0, 1].imshow(20*np.log(np.abs(lowPassFft)), cmap='gray')
plots[0, 1].set_title('low pass filtered Freq response')
plots[0, 2].imshow(np.abs(lowPassImg), cmap='gray')
plots[0, 2].set_title('low filtered Reconstructed image')

plots[1, 0].imshow(img, cmap='gray')
plots[1, 0].set_title('Original image')
plots[1, 1].imshow(20*np.log(np.abs(highPassFft)), cmap='gray')
plots[1, 1].set_title('high filtered Freq response')
plots[1, 2].imshow(np.abs(highPassImg), cmap='gray')
plots[1, 2].set_title('high filtered Reconstructed image')

fig.show()

raw_input("press any key")
