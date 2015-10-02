# Analysis of statistical algorithms for Image Classification

Various algorithms are used to do an image classification task and analyze their performance.
Various feature extraction methods are analyzed also. Finally an effort is made to develop an algorithm inspired by the visual pathway of nervous system.
## Datasets used:
1. [Natural scenes dataset](http://cvcl.mit.edu/database.htm):Each database is composed of a few hundred images of scenes belonging to the same semantic category. All of the images are in color, in jpeg format, and are 256 x 256 pixels.
2. [CIFAR-10](http://www.cs.utoronto.ca/~kriz/cifar.html): The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

## Feature extraction:
Some of the common feature extraction techniques were used. Link provoides location to the used implementation.
1. [SIFT - The Scale Invariant Feature Transform](http://www.cs.ubc.ca/~lowe/keypoints/)
2. [SURF - Speeded Up Robust Features](http://in.mathworks.com/matlabcentral/fileexchange/28300-opensurf--including-image-warp-)
3. [HOG - Histofram of Gradients](http://in.mathworks.com/matlabcentral/fileexchange/28689-hog-descriptor-for-matlab)
4. [Spatial envelope](http://people.csail.mit.edu/torralba/code/spatialenvelope/)

## Statistical models
1. GMM
2. HMM
3. SVM
4. CNN