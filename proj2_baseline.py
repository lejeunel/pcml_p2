import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import pdb
from sklearn import linear_model, svm, metrics
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedShuffleSplit
from scipy import ndimage
import glob
from natsort import natsorted

import classifier as clf
import helpers_reduced as hp

#initialize
patch_size = 16
foreground_threshold = 0.25
classifier_choice = 'rf' #SVM log_reg rf  boost
k = 5 #number of cross-validation scores
overlap = 0

# paths
data_dir = 'training'
img_dir = 'images'
gt_dir = 'groundtruth'
paths = {}
paths['train_images'] = os.path.join(data_dir, img_dir)
paths['groundtruth'] = os.path.join(data_dir, gt_dir)

datafiles_train = glob.glob(os.path.join(paths['train_images'], '*.*'))
datafiles_train = natsorted(datafiles_train)
datafiles_gt = glob.glob(os.path.join(paths['groundtruth'], '*.*'))
datafiles_gt = natsorted(datafiles_gt)

# load data and break to patches
images = [hp.load_image(datafile) for datafile in datafiles_train]
groundtruth = [hp.load_image(datafile) for datafile in datafiles_gt]

img_patches = [hp.img_crop2(img, patch_size, patch_size, overlap) for img in images]
gt_patches = [hp.img_crop2(gt, patch_size, patch_size, overlap) for gt in groundtruth]
img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

# create features and assign class for each patch
#X_6d = np.asarray([hp.extract_features(img_patches[i]) for i in range(len(img_patches))])
X_2d = np.asarray([hp.extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
Y = np.asarray([hp.value_to_class(np.mean(gt_patches[i]), foreground_threshold) for i in range(len(gt_patches))])

# choose features to try
# features = np.load('features.npy') # np.load('features_full.npy')
# Y = np.load('labels.npy') #np.load('labels_full.npy')
features = X_2d

# create classifier
classifier = clf.get_classifier(classifier_choice)
# fit classifier
classifier.fit(X_2d, Y)

# cross validate
f1_scores = cross_val_score(classifier, features, Y, cv=k, scoring='f1')
print("F1-score: %0.2f (+/- %0.2f)" % (f1_scores.mean(), f1_scores.std() * 2))

# predict on single image and overlay
# idx = 1
# img_patches_test = hp.img_crop2(images[idx], patch_size, patch_size, overlap)
# X_2d_test = np.asarray([hp.extract_features_2d(img_patches_test[i]) for i in range(len(img_patches_test))])
# Y_est = classifier.predict(X_2d_test)
# h = groundtruth[idx].shape[0]
# w = groundtruth[idx].shape[1]
# predicted_img = hp.label_to_img(h, w, patch_size, patch_size, Y_est)
# new_img = hp.make_img_overlay(images[idx], predicted_img)
# plt.imshow(new_img)
# plt.axis('off')
# plt.title('Original image overlayed by predicted mask')
# plt.show()
