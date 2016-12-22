"""
This script loads the testing images and the model that was trained on the training images,
and generates and stores the masks for the testing images.
The masks are saved under the directory 'testing_results/imageName.png'
The code generating the submission from the masks is also included
"""

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from sklearn import linear_model, svm, metrics
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedShuffleSplit
from scipy import ndimage
import scipy.misc
import glob
from natsort import natsorted

import helpers_reduced as hp
import classifier as clf
from classifier import dice_coef, dice_coef_loss, precision, recall, fbeta_score, fmeasure
exec(open('keras_imports.py').read())

#initialize
classifier_choice = 'Unet'
version = 'unet'

# load data
test_dir = 'testing'
# directory to save the predicted maps as png
test_results_dir = 'testing_results_' + version
os.makedirs(test_results_dir, exist_ok=True)

datafiles_test = []
datafiles_results = []
# loop over all the individual folders, each folder has a single test image in it
for dirpath, dirnames, filenames in os.walk(test_dir):
	for filename in filenames:
		datafiles_test.append(os.path.join(dirpath, filename))
		datafiles_results.append(os.path.join(test_results_dir, filename))

datafiles_test = natsorted(datafiles_test)
datafiles_results = natsorted(datafiles_results)

images = np.asarray([hp.load_image(datafile) for datafile in datafiles_test])
# make sure that the images are of shape (nImages, nChannels, nRows, nCols), because this is how the model was trained to have as an input
images = np.asarray([np.moveaxis(x, 2, 0) for x in images])
shape = images.shape[1:]

# create model (architecture only)
model = clf.get_classifier(classifier_choice, input_shape=shape)

# load saved model weights
model.load_weights('unet_weights.hdf5')
# load model
#model = load_model('unet.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

# predict on test images
for idx in range(len(images)):
	pred_map = model.predict(images[idx][np.newaxis,:,:,:])
	scipy.misc.imsave(datafiles_results[idx], np.squeeze(pred_map))

# make submission file
submission_filename = 'submission_' + version + '.csv'
image_filenames = glob.glob(os.path.join(test_results_dir, '*.*'))
image_filenames = natsorted(image_filenames)
hp.masks_to_submission(submission_filename, *image_filenames)
