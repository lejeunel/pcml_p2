"""
This script contains functions related to the classification
"""
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import scipy as sp
import numpy as np
import scipy.misc
import os
exec(open('keras_imports.py').read())

smooth = 1.

def get_classifier(clf, input_shape = None):
	"""
	This function returns a classifier object defined by clf
	INPUTS:
	@str		: a string indicating which classifier will be used
	@input_shape: in the case that a CNN model will be trained, the input shape
				  (it is necessary to build the model)
	OUTPUT
	@classifier : a classifier object
	"""
	if clf == 'SVM':
		classifier = svm.SVC(C=1e5, kernel='rbf', class_weight="balanced")
	elif clf == 'log_reg':
		classifier = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
	elif clf == 'rf':
		classifier = RandomForestClassifier(n_estimators=50, max_depth = 10, class_weight="balanced")
	elif clf == 'boost':
		classifier = AdaBoostClassifier()
	elif clf == 'cnn':
		# CNN network to classify patches
		input_img = Input(shape=input_shape)
		x = input_img

		x = Convolution2D(32, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)
		x = Convolution2D(32, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)

		x = Convolution2D(32, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		x = Convolution2D(64, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)
		x = Convolution2D(64, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		x = Convolution2D(128, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)
		x = Convolution2D(128, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		f = x
		x = Flatten()(x)
		x = Dense(16)(x)
		x = LeakyReLU()(x)
		x = Dropout(0.5)(x)
		x = Dense(2)(x)
		o = Activation('softmax')(x)

		# model train
		classifier = Model(input_img, o)
		classifier.summary()
		classifier.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=[fmeasure])

	elif clf == 'Unet':
		# unet network, https://github.com/jocicmarko/ultrasound-nerve-segmentation
		inputs = Input(input_shape)
		conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
		conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
		conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
		conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
		conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
		conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

		up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
		conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
		conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

		up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
		conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
		conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

		up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
		conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
		conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

		up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
		conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
		conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

		conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

		classifier = Model(input=inputs, output=conv10)
		classifier.summary()
		classifier.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=[fmeasure])
	else:
		sys.exit("The classifier you chose is not implemented")

	return classifier

def dice_coef(y_true, y_pred):
	# calculates dice coefficient
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	# calculates loss related to dice coefficient
	return -dice_coef(y_true, y_pred)

def precision(y_true, y_pred):
    # Calculates the precision, a metric for multi-label classification of
    # how many selected items are relevant.
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall, a metric for multi-label classification of
    # how many relevant items are selected.
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
	#Calculates the f-measure, the harmonic mean of precision and recall.
	return fbeta_score(y_true, y_pred, beta=1)

class EvaluateCallback(keras.callbacks.Callback):
    """
    This callback class passes an image through the network during training, and saves the
    network output (save_every defines the number of batches after which a new prediction is made).
    It serves as a monitor to see how a network slowly learns to identify the interesting structures.
    """
    def __init__(self,
                 original,
                 truth,
                 out_path=None,
                 verbose=True,
                 save_every=50):
        keras.callbacks.Callback.__init__(self)
        self.n_batch = 0
        self.n_epoch = 0
        self.out_path = out_path
        self.save_every = save_every
        self.verbose = verbose
        self.image = np.expand_dims(original, 0)
        self.truth = np.expand_dims(truth, 0)

		# create path to save intermediate results, if it does not exist
        os.makedirs(out_path, exist_ok=True)

		# 'th': image is of shape (nrows, ncols, nChannels)
        if K.image_dim_ordering() == "th":
            original = np.swapaxes(np.swapaxes(original, 0, 1), 1, 2)

        if out_path is not None:
            sp.misc.imsave(
                os.path.join(self.out_path, "original.png"),
                original)
            sp.misc.imsave(
                os.path.join(self.out_path, "truth.png"), np.squeeze(truth))

    def on_batch_end(self, batch, logs={}):
        if self.n_batch % self.save_every == 0:
            if self.verbose:
                loss = self.model.evaluate(self.image, self.truth)
                print("Current validation loss: ", loss)
                print()

            if self.out_path is not None:
                mask = self.model.predict(self.image, batch_size=1)[0]

                if self.verbose:
                    with open(
                            os.path.join(self.out_path, "loss.txt"),
                            mode="w") as f:
                        f.write(str(loss))
                        f.flush()
                sp.misc.imsave(
                    os.path.join(
                        self.out_path,
                        "{0}x{1}_mask.png".format(self.n_epoch, self.n_batch)),
                    np.squeeze(mask))
        self.n_batch += 1

    def on_epoch_end(self, batch, lgos={}):
        self.n_epoch += 1
