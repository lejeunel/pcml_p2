from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier
exec(open('keras_imports.py').read())

def get_classifier(clf, input_shape = None):
	if clf == 'SVM':
		classifier = svm.SVC(C=1e5, kernel='rbf', class_weight="balanced")
	elif clf == 'log_reg':
		classifier = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
	elif clf == 'rf':
		classifier = RandomForestClassifier(n_estimators=10, class_weight="balanced")
	elif clf == 'cnn':
		input_img = Input(shape=input_shape)
		x = input_img

		x = Convolution2D(16, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)
		x = Convolution2D(16, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)

		x = Convolution2D(32, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)
		x = Convolution2D(32, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2))(x) #2,1 for dummy small

		x = Convolution2D(48, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)
		x = Convolution2D(48, 3, 3, border_mode='same')(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2))(x) #2,1 for dummy small

		x = Convolution2D(1, 1, 1, border_mode='same')(x)
		x = LeakyReLU()(x)
		# k = 2
		# for i in range(self.params['nConv']):
		# 	x = Convolution2D((i + 1) * k, 3, 3, border_mode='same')(x)
		# 	x = LeakyReLU()(x)
		# 	#x = BatchNormalization()(x)
		# 	x = MaxPooling2D(pool_size=(2,2))(x) #2,1 for dummy small
		# 	#x = SpatialDropout2D(0.25)(x)

		f = x
		x = Flatten()(x)
		x = Dense(16)(x)
		x = LeakyReLU()(x)
		x = Dropout(0.5)(x)
		x = Dense(2)(x)
		o = Activation('softmax')(x)

		# let's train the model using SGD + momentum (how original).
		classifier = Model(input_img, o)
		classifier.summary()
		classifier.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
	else:
		classifier = 5

	return classifier
