from sklearn import (cluster,decomposition,preprocessing)
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import helpers as hp
from PIL import Image
from skimage import (morphology, feature,color,transform)
from scipy import cluster
#from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
import skimage.draw

#Parameters
n = 40 #Num of images for training
#n = min(60, len(files)) # Load maximum 20 images
grid_step = 4 #keypoints are extracted every 1/grid_scale pixels
n_im_pca = 10 #Num of images for PCA
n_components_pca = 60 #Num of components for PCA compression
sift_sigmas = np.array([1, 2, 3])
n_clusters_bow = 30 #Bag of words, num. of clusters
patch_size = 16 # each patch is 16*16 pixels
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
canny_sigma = 4 #sigma of gaussian filter prior to canny edge detector

# Loaded a set of images
root_dir = "training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
print("Loading " + str(n) + " training images")
#imgs = [color.rgb2hsv(hp.load_image(image_dir + files[i])) for i in range(n)]
imgs = [hp.load_image(image_dir + files[i]) for i in range(n)]

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " ground-truth images")
gt_imgs = [hp.load_image(gt_dir + files[i]) for i in range(n)]

print('Image size = ' + str(imgs[0].shape[0]) + ',' + str(imgs[0].shape[1]))

#features are extracted from n_im_pca images on a dense grid (grid_step). PCA is then applied for dimensionality reduction.
print('Extracting features (SIFT, RGB, etc..)')
X_pca = hp.get_features(imgs[0:n_im_pca],canny_sigma,sift_sigmas,grid_step)
print('Extracting euclidean distance transform')
X_dt = hp.get_features_dt(imgs[0:n_im_pca],canny_sigma,grid_step)

print('Scaling PCA input')
scaler = preprocessing.StandardScaler().fit(X_pca)
X_pca = scaler.transform(X_pca)

print('Fitting PCA model with n_components = ' + str(n_components_pca))
pca = decomposition.PCA(n_components_pca)
pca.fit(X_pca)
#Plot explained variance
plt.figure()
plt.plot(pca.explained_variance_, linewidth=2)
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.show()
X_pca_transf = pca.transform(X_pca)
X_to_cluster = np.concatenate((X_pca_transf, X_dt),axis=1)

import pdb; pdb.set_trace()
print('Generating codebook on ' + str(n_im_pca) + ' images with ' + str(n_clusters_bow) + ' clusters')
codebook, distortion = cluster.vq.kmeans(X_to_cluster, n_clusters_bow,thresh=1)

codes = list()
print('Generating codes on ' + str(n) + ' images')
for i in range(n):
    sys.stdout.write('\r')
    this_x_sift = hp.get_features([imgs[i]],canny_sigma,sift_sigmas,1)
    this_x_dt = hp.get_features_dt([imgs[i]],canny_sigma,1)
    this_x_sift = scaler.transform(this_x_sift)
    this_x_transf = pca.transform(this_x_sift)
    this_x = np.concatenate((this_x_transf,this_x_dt),axis=1)
    code, dist = cluster.vq.vq(this_x, codebook)
    code = code.reshape(imgs[0].shape[0],imgs[0].shape[1])
    sys.stdout.write("%d/%d" % (i+1, n))
    sys.stdout.flush()
    codes.append(code)
    #histogram_of_words, bin_edges = np.histogram(code,bins=range(codebook.shape[0] + 1),normed=True)

import pdb; pdb.set_trace()
hist_bow = list()
print('Getting Bag-of-Visual-Words on patches of size ' + str(patch_size))
hist_bins = n_components_pca

for i in range(len(codes)):
    code_patch = hp.img_crop(codes[i], patch_size, patch_size)
    hist_bow.append([np.histogram(code_patch[i],bins=hist_bins)[0] for i in range(len(code_patch))])

#hough_patches = [hp.img_crop(hough_lines[i], patch_size, patch_size) for i in range(n)]
img_patches = [hp.img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
gt_patches = [hp.img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

# Linearize list of patches
print('Linearizing list of patches')
bow_patches = np.asarray([hist_bow[i][j] for i in range(len(hist_bow)) for j in range(len(hist_bow[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

# Compute features for each image patch
print('Computing features')
X = bow_patches
Y = np.asarray([hp.value_to_class(np.mean(gt_patches[i]),foreground_threshold) for i in range(len(gt_patches))])

print('Rebalance classes by oversampling')
n_pos_to_add = np.sum(Y==0) - np.sum(Y==1)
idx_to_duplicate = np.where(Y==1)[0][np.random.randint(0,np.sum(Y==1),n_pos_to_add)]
X = np.concatenate((X,X[idx_to_duplicate,:]),axis=0)
Y = np.concatenate((Y,Y[idx_to_duplicate]),axis=0)

#plt.scatter(X_mean_var[:, 0], X_mean_var[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired); plt.show()

#Print feature statistics
print('Computed ' + str(X.shape[0]) + ' features')
print('Feature dimension = ' + str(X.shape[1]))
print('Number of classes = ' + str(np.max(Y)))

Y0 = [i for i, j in enumerate(Y) if j == 0]
Y1 = [i for i, j in enumerate(Y) if j == 1]
print('Class 0: ' + str(len(Y0)) + ' samples')
print('Class 1: ' + str(len(Y1)) + ' samples')

# train a logistic regression classifier
from sklearn import (linear_model,svm,preprocessing)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#print('Scaling training data to mean 0 and variance 1')
#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)

print('Training SVM classifier')
my_svm = svm.LinearSVC()
my_svm.fit(X,Y)

print('Training Adaboost classifier')
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), algorithm="SAMME", n_estimators=1000)
bdt.fit(X,Y)

# we create an instance of the classifier and fit the data
print('Training LogReg classifier')
logreg = linear_model.LogisticRegression(C=1)
#logreg = linear_model.LogisticRegression(C=1, class_weight="balanced")
logreg.fit(X, Y)

print('Training Ridge classifier')
linreg = linear_model.LinearRegression()
#linreg = linear_model.Ridge(alpha=1.0)
linreg.fit(X, Y)

for the_classifier in {linreg,logreg,my_svm,bdt}:

    # Predict on the training set
    Z = the_classifier.predict(X)
    thr_pred = 0.5
    #thr_pred = np.mean(Z)
    #thr_pred = 0

    # Get non-zeros in prediction and grountruth arrays
    Z_pos = np.where(Z>=thr_pred)[0]
    Z_neg = np.where(Z<thr_pred)[0]
    Y_pos = np.nonzero(Y)[0]
    Y_neg = np.where(Y==0)[0]

    TPR = len(list(set(Y_pos) & set(Z_pos))) / float(len(Z))
    FPR = len(list(set(Y_neg) & set(Z_pos))) / float(len(Z))
    print('Results with classifier: ' + the_classifier.__class__.__name__)
    print('TPR/FPR = ' + str(TPR) + '/' + str(FPR))


my_classifier = bdt
# Run prediction on the img_idx-th image
img_idx = 10
the_img = imgs[img_idx]
the_gt = gt_imgs[img_idx]

this_x_sift = hp.get_features([the_img],canny_sigma,sift_sigmas,1)
this_x_dt = hp.get_features_dt([the_img],canny_sigma,1)
this_x_sift = scaler.transform(this_x_sift)
this_x_transf = pca.transform(this_x_sift)
this_x = np.concatenate((this_x_transf,this_x_dt),axis=1)
code, dist = cluster.vq.vq(this_x, codebook)
code = code.reshape(the_img.shape[0],the_img.shape[1])

code_patch = hp.img_crop(code, patch_size, patch_size)
hist_bow = np.asarray([np.histogram(code_patch[i],bins=hist_bins)[0] for i in range(len(code_patch))])

Xi = hist_bow

#Xi = scaler.transform(Xi)
Zi = my_classifier.predict(hist_bow)

# Display prediction as an image
w = gt_imgs[img_idx].shape[0]
h = gt_imgs[img_idx].shape[1]
predicted_im = hp.label_to_img(w, h, patch_size, patch_size, Zi)
#img_overlay = hp.make_img_overlay(the_img, predicted_im)
img_overlay = color.label2rgb(predicted_im,the_img,alpha=0.5)
img_over_conc = hp.concatenate_images(img_overlay,color.gray2rgb(gt_imgs[img_idx]))
fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size
plt.imshow(img_over_conc, cmap='Greys_r');
plt.title('Pred of im ' + str(img_idx) + ' with ' +  my_classifier.__class__.__name__)
plt.show()
