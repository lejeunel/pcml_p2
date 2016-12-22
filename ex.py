import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
#import Image
#from PIL.Image import core as image
import pdb
from sklearn import linear_model, svm, metrics
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedShuffleSplit
#from skimage.filters import gabor_kernel
#from skimage.transform import rescale, rotate
from scipy import ndimage
import scipy.misc
import glob
from natsort import natsorted

import helpers_reduced as hp
#import helpers as hp
import classifier as clf
from classifier import dice_coef, dice_coef_loss, precision, recall, fbeta_score, fmeasure
exec(open('keras_imports.py').read())

#initialize
classifier_choice = 'Unet'
version = 'unet_final'

# load data
images = np.load('images.npy')
image = images[1,:,:,:]
shape = images.shape[1:]

# create model (architecture only)
model = clf.get_classifier(classifier_choice, input_shape=shape)

# load saved model weights
model.load_weights('unet2_weights_final.hdf5')
# load model
#model = load_model('unet.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

# predict on test images

pred_map = model.predict(image[np.newaxis,:,:,:])

new_img = hp.make_img_overlay(np.moveaxis(image,0,2), np.squeeze(pred_map))

pdb.set_trace()
plt.imshow(new_img)
plt.show()
