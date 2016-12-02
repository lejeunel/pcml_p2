from sklearn import (cluster,decomposition)
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
grid_scale = 0.25 #keypoints are extracted every 1/grid_scale pixels
n_im_pca = 10 #Num of images for PCA
n_components_pca = 50 #Num of components for PCA compression
n_clusters_bow = 10 #Bag of words, num. of clusters
patch_size = 16 # each patch is 16*16 pixels
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# Loaded a set of images
root_dir = "training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
print("Loading " + str(n) + " training images")
imgs = [color.rgb2hsv(hp.load_image(image_dir + files[i])) for i in range(n)]

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " ground-truth images")
gt_imgs = [hp.load_image(gt_dir + files[i]) for i in range(n)]

print('Image size = ' + str(imgs[0].shape[0]) + ',' + str(imgs[0].shape[1]))

#SIFT and RGB features are extracted from n_im_pca images on a dense grid (grid_scale). PCA is then applied for dimensionality reduction.
print('computing PCA on ' + str(n_im_pca) + ' images')
print('computing features (SIFT and RGB)')
X_sift = np.asarray([ hp.get_sift_densely(imgs[i],step=int(patch_size*grid_scale)) for i in range(n_im_pca)])
X_sift = X_sift.reshape(n_im_pca*X_sift.shape[1],X_sift.shape[2])
X_rgb = np.asarray([ transform.rescale(imgs[i],scale=grid_scale) for i in range(n_im_pca)])
X_rgb = X_rgb.reshape(n_im_pca*X_rgb.shape[1]*X_rgb.shape[2],X_rgb.shape[3])
X_pca = np.concatenate((X_sift,X_rgb),axis=1)

print('Computing SVD with n_components = ' + str(n_components_pca))
pca = decomposition.PCA()
pca.fit(X_pca)
#Plot explained variance
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.show()
pca.n_components = n_components_pca
X_pca_transf = pca.transform(X_pca)


print('Generating codebook on ' + str(n_im_pca) + ' images with ' + str(n_clusters_bow) + ' clusters')
codebook, distortion = cluster.vq.kmeans(X_pca_transf, n_clusters_bow,thresh=1)

codes = list()
print('Generating codes on ' + str(n) + ' images')
for i in range(n):
    im_shape = imgs[i].shape
    this_im_sift = hp.get_sift_densely(imgs[i],step=1)
    this_im = imgs[i].reshape(imgs[i].shape[0]*imgs[i].shape[1],imgs[i].shape[2])
    this_x = np.concatenate((this_im_sift,this_im),axis=1)
    this_x_transf = pca.transform(this_x)
    code, dist = cluster.vq.vq(this_x_transf, codebook)
    code = code.reshape(im_shape[0],im_shape[1])
    codes.append(code)
    #histogram_of_words, bin_edges = np.histogram(code,bins=range(codebook.shape[0] + 1),normed=True)

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

my_classifier = my_svm #This points to one of {bdt,my_svm,logreg,linreg}
# Predict on the training set
Z = my_classifier.predict(X)

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
print('Results with classifier: ' + my_classifier.__class__.__name__)
print('TPR/FPR = ' + str(TPR) + '/' + str(FPR))

# Run prediction on the img_idx-th image
img_idx = 10
the_img = imgs[img_idx]

im_shape = the_img.shape
this_im_sift = hp.get_sift_densely(the_img,step=1)
this_im = the_img.reshape(the_img.shape[0]*the_img.shape[1],the_img.shape[2])
this_x = np.concatenate((this_im_sift,this_im),axis=1)
this_x_transf = pca.transform(this_x)
code, dist = cluster.vq.vq(this_x_transf, codebook)
code = code.reshape(im_shape[0],im_shape[1])

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
img_overlay = color.label2rgb(predicted_im,the_img)
img_over_conc = hp.concatenate_images(img_overlay,color.gray2rgb(gt_imgs[img_idx]))
fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size
plt.imshow(img_over_conc, cmap='Greys_r');
plt.title('Prediction. ' + my_classifier.__class__.__name__)
plt.show()
