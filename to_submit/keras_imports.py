# imports related to keras framework
from keras.models import Model
from keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Flatten, UpSampling2D
from keras.layers import Convolution2D, MaxPooling2D, LeakyReLU, GaussianNoise, ZeroPadding2D
from keras.layers import Input, merge
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
import keras
K.set_image_dim_ordering('th')
