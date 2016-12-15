import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import pdb
from sklearn import linear_model, svm, metrics
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedShuffleSplit
from skimage.filters import gabor_kernel
from skimage.transform import rescale, rotate
from scipy import ndimage
import glob
from natsort import natsorted

import classifier as clf
import helpers as hp

#initialize
patch_sizes = [16]
foreground_threshold = 0.25
classifier_choice = 'rf' #SVM log_reg
k = 5 #number of cross-validation scores
overlap = 0

# paths
projectPath = '/home/tat/mnt/OTL/1-Projects/OCT_SRT/project2'
data_dir = 'training'
img_dir = 'images'
gt_dir = 'groundtruth'
paths = {}
paths['train_images'] = os.path.join(projectPath, data_dir, img_dir)
paths['groundtruth'] = os.path.join(projectPath, data_dir, gt_dir)

datafiles_train = glob.glob(os.path.join(paths['train_images'], '*.*'))
datafiles_train = natsorted(datafiles_train)
datafiles_gt = glob.glob(os.path.join(paths['groundtruth'], '*.*'))
datafiles_gt = natsorted(datafiles_gt)
nSamples = len(datafiles_train)

# load data and break to patches
images = [hp.load_image(datafile) for datafile in datafiles_train]
groundtruth = [hp.load_image(datafile) for datafile in datafiles_gt]

for patch_size in patch_sizes:
	print('Patch size:' + str(patch_size))
	img_patches = [hp.img_crop2(img, patch_size, patch_size, overlap) for img in images]
	gt_patches = [hp.img_crop2(gt, patch_size, patch_size, overlap) for gt in groundtruth]
	img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
	img_patches_2d = np.asarray([np.mean(x, axis=2) for x in img_patches])
	gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

	#augment data

	# gabor filters
	# theta = range(1,4)
	# sigma = (1, 3)
	# freq = (0.05, 0.25)

	#gabor_kernels = hp.get_gabor_kernels(theta, sigma, freq)
	# convolve kernels with data and use filter values as feature vectors!
	# gabor_features = np.asarray([np.squeeze(hp.compute_feats(patch, gabor_kernels)) for patch in img_patches_2d])
	# show kernels
	# fig = plt.figure()
	# plt.title('Gabor kernels')
	# plt.axis('off')
	# for i in range(len(gabor_kernels)):
	# 	fig.add_subplot(4,3,i)
	# 	plt.imshow(gabor_kernels[i], cmap = 'gray')
	# 	plt.axis('off')
	# plt.show()

	#show data
	# for i in range(nSamples):
	# 	fig = plt.figure()
	# 	fig.add_subplot(1,2,1)
	# 	plt.imshow(images[i])
	# 	plt.axis('off')
	# 	fig.add_subplot(1,2,2)
	# 	plt.imshow(groundtruth[i], cmap = 'gray')
	# 	plt.axis('off')
	# 	plt.show()
	# 	pdb.set_trace()

	# create features and assign class for each patch
	X_2d = np.asarray([hp.extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
	#X_6d = np.asarray([hp.extract_features(img_patches[i]) for i in range(len(img_patches))])
	Y = np.asarray([hp.value_to_class(np.mean(gt_patches[i]), foreground_threshold) for i in range(len(gt_patches))])

	# choose features to try
	features = X_2d

	# Print feature statistics
	print('Computed ' + str(features.shape[0]) + ' features')
	print('Feature dimension = ' + str(features.shape[1]))

	Y0 = [i for i, j in enumerate(Y) if j == 0]
	Y1 = [i for i, j in enumerate(Y) if j == 1]
	print('Class 0: ' + str(len(Y0)) + ' samples')
	print('Class 1: ' + str(len(Y1)) + ' samples')

	# Plot 2d features using groundtruth to color the datapoints
	# plt.scatter(X_2d[:, 0], X_2d[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
	# plt.show()

	# Plot 2d features for each color using groundtruth to color the datapoints
	# it seems that every color has more or less the same distribution as the mean
	# fig = plt.figure()
	# colors = ['Red', 'Green', 'Blue']
	# for i in range(3):
	# 	fig.add_subplot(3,1,i)
	# 	plt.title(colors[i])
	# 	plt.scatter(X_6d[:, i], X_6d[:, i+3], c=Y, edgecolors='k', cmap=plt.cm.Paired)
	# plt.show()

	# create classifier
	classifier = clf.get_classifier(classifier_choice)

	# grid hyperparameter search
	# param_grid = [
	#   {'C': [1, 1000, 1e5, 1e7], 'kernel': ['poly'], 'degree': [3,4,5,6], 'class_weight': ['balanced']},
	#   {'C': [1,  1000, 1e5, 1e7], 'gamma': [0.001, 0.0001], 'kernel': ['rbf'], 'class_weight': ['balanced']},
	#  ]
	# cv = StratifiedShuffleSplit(n_splits=k, test_size=0.2, random_state=42)
	# # grid search of parameters
	# grid = GridSearchCV(classifier, param_grid, scoring='f1', cv=cv)
	# grid.fit( X_2d, Y)
	"""
	# train classifier
	classifier.fit(X_2d, Y)
	# predict labels of training set
	Y_est = classifier.predict(X_2d)
	# calculate F score
	f1 = metrics.f1_score(Y, Y_est)
	print('F1-score = ' + str(f1))
	"""

	# cross validate
	f1_scores = cross_val_score(classifier, features, Y, cv=k, scoring='f1')
	print("F1-score: %0.2f (+/- %0.2f)" % (f1_scores.mean(), f1_scores.std() * 2))

pdb.set_trace()
print('eo')
