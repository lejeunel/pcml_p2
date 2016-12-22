# Pattern recognition and machine learning, project 2. Laurent Lejeune, Tatiana Fountoukidou, Guillaume de Montauzon

##Requirements
- numpy
- matplotlib
- scikit-learn
- scikit-image
- scipy
- PIL (or Pillow)
- Keras
- natsort
- pystruct (https://pystruct.github.io/). This package depends on cvxopt for convex optimization (available on pip)
- opencv. Since we make use of SIFT, it is necessary to compile the library using the non-free modules available here: https://github.com/opencv/opencv_contrib. A guide for compiling those modules: http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/

##Files descriptions
- train_unet.py			: trains a Unet CNN network and saves the weights
- run.py 				: loads trained model, applies on testing set and saves predictions in folder "testing_results", generates the final submission file
- run_refinements.py: Trains and cross-validates logreg, random-forest, refined-logreg, and refined-random-forest
- run_baseline.py 	: runs a random forest classifier on features, either 2d(mean and variance) or 65d(extracted features), 5-fold cross validation
- classifier.py 		: auxiliary file with functions connected to the classification task
- keras_imports.py 		: imports related to the keras framework
- helpers_reduced.py 	: helper functions used in run_baseline.py
- helpers.py 			: helper functions usd in run_refinements.py
- augment.py			: extension of keras image generator to augment the groundtruth 
						  (https://www.kaggle.com/hexietufts/ultrasound-nerve-segmentation/easy-to-use-keras-imagedatagenerator/code)

##Data files

### On external Dropbox storage
Due to the large size of these files, we chose to store it here (open access): https://www.dropbox.com/sh/smlbwzble5bqob7/AABNF5hPAahKWDrJpPbnMbeba?dl=0
- unet_weights.hdf5 	: HDF5 file with the trained weights for the unet network
- training: Directory containing provided training images and ground-truths
- testing: Directory containing provided testing images
### Feature files
Those files are not necessary, the run_refinements.py will run fine without those
- features.npy			: numpy matrix of superpixel feature vectors for all the images 
- labels.npy			: numpy array with label for each superpixel feature vector
