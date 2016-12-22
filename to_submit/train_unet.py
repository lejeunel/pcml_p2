import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from sklearn import linear_model, svm, metrics
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedShuffleSplit
from scipy import ndimage
import glob
from natsort import natsorted

import classifier as clf
import augment as aug
import helpers_reduced as hp
exec(open('keras_imports.py').read())
sys.setrecursionlimit(100000)

#initialize
classifier_choice = 'Unet' #SVM log_reg Unet cnn
#batch size
bs = 8

# load data
# images = np.load('images.npy')
# groundtruth = np.load('gt_masks.npy')
data_dir = 'training'
img_dir = 'images'
gt_dir = 'groundtruth'
paths = {}
paths['train_images'] = os.path.join(data_dir, img_dir)
paths['groundtruth'] = os.path.join(data_dir, gt_dir)

# glob returns the filenames in random order, so we perform natural sort
# to make sure each groundtruth corresponds to the correct image
datafiles_train = glob.glob(os.path.join(paths['train_images'], '*.*'))
datafiles_train = natsorted(datafiles_train)
datafiles_gt = glob.glob(os.path.join(paths['groundtruth'], '*.*'))
datafiles_gt = natsorted(datafiles_gt)

images = np.asarray([hp.load_image(datafile) for datafile in datafiles_train])
groundtruth = np.asarray([hp.load_image(datafile) for datafile in datafiles_gt])
# change image shape from (nChannels, nrows, ncols) to (nrows, ncols, nChannels)
images = np.asarray([np.moveaxis(x, 2,0) for x in images])

# divide the training data into training and validation
images_val = images[0:10,:,:,:]
images_train = images[10:,:,:,:]

groundtruth = groundtruth[:,np.newaxis,:,:]
# binarize groundtruth
zero_ind = groundtruth < 0.5
one_ind = groundtruth >= 0.5
groundtruth[zero_ind] = 0
groundtruth[one_ind] = 1
test_gt = groundtruth[0]
groundtruth_val = groundtruth[0:10,:,:,:]
groundtruth_train = groundtruth[10:,:,:,:]

shape = images.shape[1:]

#augment data
# augment training set
train_gen = aug.ImageDataGenerator(featurewise_center=False,
	samplewise_center=False,
	featurewise_std_normalization=False,
	samplewise_std_normalization=False,
	zca_whitening=False,
	rotation_range=40,
	width_shift_range=0,
	height_shift_range=0,
	shear_range=0.,
	zoom_range=0,
	channel_shift_range=0.,
	fill_mode='nearest',
	cval=0.,
	horizontal_flip=True,
	vertical_flip=True,
	rescale=1.4,
	dim_ordering=K.image_dim_ordering())

# generator of validation set - no augmentation
val_gen = ImageDataGenerator(featurewise_center=False,
	samplewise_center=False,
	featurewise_std_normalization=False,
	samplewise_std_normalization=False,
	zca_whitening=False,
	rotation_range=0,
	width_shift_range=0,
	height_shift_range=0,
	shear_range=0.,
	zoom_range=0,
	channel_shift_range=0.,
	fill_mode='nearest',
	cval=0.,
	horizontal_flip=False,
	vertical_flip=False,
	rescale=None,
	dim_ordering=K.image_dim_ordering())

#create classifier
classifier = clf.get_classifier(classifier_choice, input_shape=shape)
# define callbacks
# save intermediate training steps (every 50 batches)
inter = clf.EvaluateCallback(images_val[0], groundtruth_val[0], out_path = 'inter_test_image', verbose = False)
# save model instances with lowest validation loss
model_checkpoint = ModelCheckpoint('unet_weights.hdf5', monitor='val_loss', verbose =1, save_best_only=True, save_weights_only = True)
# early stop if validation loss does not significantly drop after 10 epochs. Saves time and
# avoids overfitting
early_stop = EarlyStopping(patience = 10)
# classifier.fit(images, groundtruth, batch_size = 16, validation_split = 0.1, nb_epoch=100, callbacks=[model_checkpoint, stef])
hist = classifier.fit_generator(train_gen.flow(images_train, groundtruth_train, batch_size=bs),
						samples_per_epoch=2000,
						nb_epoch=100,
						validation_data = val_gen.flow(images_val, groundtruth_val),
						nb_val_samples = len(images_val),
						callbacks=[model_checkpoint, inter, early_stop])

# also save weights after last epoch, since loss is binary crossentropy
# and maybe not equal to f-score optimization, and choose the version with
# the highest validation f-score afterwards
classifier.save_weights('unet_weights_final.hdf5')

np.save('train_loss', hist.history['loss'])
np.save('val_loss', hist.history['val_loss'])
