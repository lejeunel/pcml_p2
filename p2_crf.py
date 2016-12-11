import re
from sklearn import (cluster,decomposition,preprocessing,utils,metrics)
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import helpers as hp
from PIL import Image
from skimage import (morphology, feature,color,transform,filters)
from scipy import cluster
#from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
from skimage import (draw,transform)

#Parameters
n = 40 #Num of images for training
#n = min(60, len(files)) # Load maximum 20 images
grid_step = 16 #keypoints are extracted every grid_step pixels
#n_im_pca = 20 #Num of images for PCA
n_im_pca = 10 #Num of images for PCA
n_components_pca = 60 #Num of components for PCA compression
#sift_sigmas = np.array([1, 2, 3])
sift_sigmas = None
n_clusters_bow = 500 #Bag of words, num. of clusters
patch_size = 16 # each patch is 16*16 pixels
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
canny_sigma = 4 #sigma of gaussian filter prior to canny edge detector
n_colors = 128
hough_max_lines = 50

outliers = np.array([10,20,26,27,64,76])

# Loaded a set of images
root_dir = "training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))

files_clean = list()
for i in range(len(files)):
    if( i not in outliers):
        files_clean.append(files[i])

files = files_clean

print("Loading " + str(n) + " training images")
#imgs = [color.rgb2hsv(hp.load_image(image_dir + files[i])) for i in range(n)]
#imgs = [hp.load_image(image_dir + files[i]) for i in range(n)]
imgs = [hp.load_image(image_dir + files[i]) for i in range(len(files))]

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " ground-truth images")
#gt_imgs = [hp.load_image(gt_dir + files[i]) for i in range(n)]
gt_imgs = [hp.load_image(gt_dir + files[i]) for i in range(len(files))]

print('Image size = ' + str(imgs[0].shape[0]) + ',' + str(imgs[0].shape[1]))

from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
#plt.imshow(hough_img); plt.title('thresholding, hough transform, binary dilation'); plt.show()
idx = 11

the_img = imgs[idx]
the_gt = gt_imgs[idx]

n_im_kmeans = 5
#features are extracted from n_im_pca images on a dense grid (grid_step). PCA is then applied for dimensionality reduction.
print('Extracting RGB histograms')
X_hist = hp.get_features_hist(imgs[0:n_im_kmeans],10,grid_step)
print('Extracting euclidean distance transform')
X_dt = hp.get_features_dt(imgs[0:n_im_kmeans],canny_sigma,grid_step)
print('Extracting SIFT features')
X_sift = hp.get_features_sift(imgs[0:n_im_kmeans],canny_sigma,sift_sigmas,grid_step)

print('Fitting PCA model on SIFT with n_components = ' + str(n_components_pca))
pca = decomposition.PCA(n_components_pca)
pca.fit(X_sift)
X_sift = pca.transform(X_sift)
X = np.concatenate((X_sift, X_hist,X_dt),axis=1)

print('Generating codebook on ' + str(X.shape[0]) + ' samples  with ' + str(n_clusters_bow) + ' clusters')
codebook, distortion = cluster.vq.kmeans(X, n_clusters_bow,thresh=1)

import pdb; pdb.set_trace()
codes = list()
print('Generating codes on ' + str(n) + ' images')
for i in range(n):
    sys.stdout.write('\r')
    labels1 = segmentation.slic(imgs[i], compactness=30, n_segments=slic_segments)
    this_x_sift = hp.get_features_sift([imgs[i]],canny_sigma,sift_sigmas,grid_step)
    this_x_sift = pca.transform(this_x_sift)
    this_x_dt = hp.get_features_dt([imgs[i]],canny_sigma,grid_step)
    this_x_hist = hp.get_features_hist([imgs[i]],10,grid_step)
    this_x = np.concatenate((this_x_sift, this_x_hist,this_x_dt),axis=1)
    code, dist = cluster.vq.vq(this_x, codebook)
    code = code.reshape(int(imgs[0].shape[0]/grid_step),int(imgs[0].shape[1]/grid_step))
    code = transform.rescale(code,grid_step,preserve_range=True)
    sys.stdout.write("%d/%d" % (i+1, n))
    sys.stdout.flush()
    codes.append(code)
sys.stdout.write('\n')
    #histogram_of_words, bin_edges = np.histogram(code,bins=range(codebook.shape[0] + 1),normed=True)

hist_bow = list()
print('Getting Bag-of-Visual-Words on patches of size ' + str(patch_size))
hist_bins = n_clusters_bow

for i in range(len(codes)):
    code_patch = hp.img_crop(codes[i], patch_size, patch_size)
    hist_bow.append([np.histogram(code_patch[i],bins=hist_bins)[0] for i in range(len(code_patch))])


gt_patches = [hp.img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

print('Linearizing list of patches')
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

# Compute features for each image patch
print('Building ground-truth array')
Y = np.asarray([hp.value_to_class(np.mean(gt_patches[i]),foreground_threshold) for i in range(len(gt_patches))])

print('Rebalance classes by oversampling')
n_pos_to_add = (np.sum(Y==0) - np.sum(Y==1))
idx_to_duplicate = np.where(Y==1)[0][np.random.randint(0,np.sum(Y==1),n_pos_to_add)]
X = np.concatenate((X,X[idx_to_duplicate,:]),axis=0)
Y = np.concatenate((Y,Y[idx_to_duplicate]),axis=0)

#plt.scatter(X_mean_var[:, 0], X_mean_var[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired); plt.show()

#Print feature statistics
print('Computed ' + str(X.shape[0]) + ' features')
print('Feature dimension = ' + str(X.shape[1]))
print('Number of classes = ' + str(np.max(Y)+1))

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

#print('Training SVM classifier')
#my_svm = svm.SVC(kernel='rbf')
#my_svm.fit(X,Y)

# we create an instance of the classifier and fit the data
print('Training LogReg classifier')
logreg = linear_model.LogisticRegression(C=1)
#logreg = linear_model.LogisticRegression(C=1, class_weight="balanced")
logreg.fit(X, Y)

print('Training Ridge classifier')
linreg = linear_model.LinearRegression()
#linreg = linear_model.Ridge(alpha=1.0)
linreg.fit(X, Y)

print('Training Adaboost classifier')
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=500)
bdt.fit(X,Y)

for the_classifier in {linreg,logreg,bdt}:
#for the_classifier in {linreg,logreg}:

    # Predict on the training set
    Z = the_classifier.predict(X)
    thr_pred = 0.5

    # Get non-zeros in prediction and grountruth arrays
    Z_bin = np.zeros(Z.shape)
    Z_bin[np.where(Z>=thr_pred)[0]] = 1
    Z_bin[np.where(Z<thr_pred)[0]] = 0
    Z_bin = Z_bin.astype(float)

    f1_score = metrics.f1_score(Y, Z_bin, average='weighted')
    conf_mat = metrics.confusion_matrix(Y,Z_bin)
    TPR = conf_mat[0][0]/(conf_mat[0][0] + conf_mat[0][1])
    FPR = conf_mat[1][0]/(conf_mat[1][0] + conf_mat[1][1])
    print('Results with classifier: ' + the_classifier.__class__.__name__)
    print('TPR/FPR = ' + str(TPR) + '/' + str(FPR))
    print('F1-score = ' + str(f1_score))

my_classifier = logreg
# Run prediction on the img_idx-th image
img_idx = 10
the_img = imgs[img_idx]
the_gt = gt_imgs[img_idx]

X_dt = hp.get_features_dt([imgs[img_idx]],canny_sigma,grid_step)
X_hist = hp.get_features_hist([imgs[img_idx]],10,grid_step)
X_hough = hp.get_features_hough([imgs[img_idx]],0.4,hough_max_lines,grid_step,canny_sigma,radius=1,threshold=10,line_length=45,line_gap=3)
X_sift = hp.get_features_sift([imgs[img_idx]],canny_sigma,sift_sigmas,grid_step)
pca = decomposition.PCA(n_components_pca)
pca.fit(X_sift)
X_sift = pca.transform(X_sift)
Xi = np.concatenate((X_sift, X_hough,X_hist,X_dt),axis=1)

the_gt_patches = hp.img_crop(the_gt, grid_step, grid_step)
Yi = np.asarray([hp.value_to_class(the_gt_patches[i],foreground_threshold) for i in range(len(the_gt_patches))])
Zi = my_classifier.predict(Xi)


conf_mat = metrics.confusion_matrix(Yi,Zi)
TPR = conf_mat[0][0]/(conf_mat[0][0] + conf_mat[0][1])
FPR = conf_mat[1][0]/(conf_mat[1][0] + conf_mat[1][1])
print('TPR/FPR = ' + str(TPR) + '/' + str(FPR))

# Display prediction as an image
w = gt_imgs[img_idx].shape[0]
h = gt_imgs[img_idx].shape[1]
predicted_im = hp.label_to_img(w, h, patch_size, patch_size, Zi)

img_overlay = color.label2rgb(predicted_im,the_img,alpha=0.5)
img_over_conc = hp.concatenate_images(img_overlay,color.gray2rgb(gt_imgs[img_idx]))
predicted_labels = hp.concatenate_images(img_overlay,color.gray2rgb(predicted_im))
fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size
plt.imshow(img_over_conc, cmap='Greys_r');
plt.title('im ' + str(img_idx) + ', ' +  my_classifier.__class__.__name__ + 'TPR/FPR = ' + str(TPR) + '/' + str(FPR))
plt.show()
