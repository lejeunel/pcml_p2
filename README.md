# Pattern recognition and machine learning, project 1. Laurent Lejeune, Tatiana Fountoukidou, Guillaume de Montauzon

##Requirements
The required packages are listed here:
- numpy
- matplotlib
- scikit-learn
- scikit-image
- scipy
- PIL (or Pillow)
- TensorFlow
- pystruct (https://pystruct.github.io/). This package depends on cvxopt for convex optimization (available on pip)
- opencv. Since we make use of SIFT, it is necessary to compile the library using the non-free modules available here: https://github.com/opencv/opencv_contrib. A guide for compiling those modules: http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/

## Content of this archive
- run.py: Main file used to
  - Load training and testing data from original csv files
  - Run k-fold cross-validation on various models
  - Train model and write submission csv file
- helpers.py: toolbox functions for feature extraction, CRF graph construction, ...

## Computation time (on intel i5-6300 laptop)
- Feature extraction on 94 images: ~5 minutes
- CRF graph construction on 94 images: ~5 minutes