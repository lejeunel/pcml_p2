from __future__ import absolute_import
from __future__ import print_function
#from __future__ import division

import os
import ipdb
import pdb
from pdb import set_trace as bp


import sys
import os
import time
import glob
import math
import pickle
import h5py
#import cv2

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
#import plotly.plotly as py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from natsort import natsorted, ns
from keras.utils import np_utils
from sklearn.metrics import roc_curve, auc

import random
#import theano.tensor as T
import theano
#import theano.tensor as T
#theano.sandbox.cuda.use('gpu0')

from shutil import copy

#from Pillow import Image
