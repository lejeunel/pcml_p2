import re
from sklearn import (cluster,decomposition,preprocessing,utils,metrics)
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import helpers as hp
from PIL import Image
from skimage import (morphology, feature,color,transform,filters)
from skimage import (draw,transform,data, segmentation, color)
from scipy import cluster
from skimage.future import graph
from matplotlib import pyplot as plt
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import OneSlackSSVM

r1 = np.asarray([0, 0]);
r2 = np.asarray([1, 1]);
r = r1-r2
theta = np.arctan2(r[0],r[1])
print(np.max((np.abs(np.cos(theta)),np.abs(np.sin(theta)))))

#Parameters
params = dict()
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
hough_line_gap = 3
hough_line_length = 100
hough_threshold = 1
hough_canny = 1
hough_radius = 20
hough_rel_thr = 0.5
slic_segments = 400
slic_compactness = 30

params = locals()

outliers = np.array([10,20,26,27,64,76])

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
imgs = [hp.load_image(image_dir + files[i]) for i in range(len(files))]

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " ground-truth images")
gt_imgs = [hp.load_image(gt_dir + files[i]) for i in range(len(files))]

print('Image size = ' + str(imgs[0].shape[0]) + ',' + str(imgs[0].shape[1]))

idx = 11

the_img = imgs[idx]
the_gt = gt_imgs[idx]

img_and_gt = color.label2rgb(the_gt.astype(bool),the_img)
hough = hp.make_hough(the_img,0.5,25,sig_canny=1,radius=20,threshold=1,line_length=100,line_gap=3)
hough_img = hp.concatenate_images(img_and_gt,color.rgb2gray(hough))
plt.imshow(hough_img); plt.title('thresholding, hough transform, binary dilation'); plt.show()

#features are extracted from n_im_pca images on a dense grid (grid_step). PCA is then applied for dimensionality reduction.
print('Extracting SIFT features')
X_sift = hp.get_features_sift(imgs[0:n_im_pca],canny_sigma,sift_sigmas,grid_step)

print('Fitting PCA model on SIFT with n_components = ' + str(n_components_pca))
pca = decomposition.PCA(n_components_pca)
pca.fit(X_sift)
X_sift = pca.transform(X_sift)

print('Generating SIFT codebook on ' + str(X_sift.shape[0]) + ' samples  with ' + str(n_components_pca) + ' clusters')
codebook, distortion = cluster.vq.kmeans(X_sift, n_components_pca,thresh=1)

print("Extracting features on " + str(n) + " images")
X = list()
sp_labels = list()
for i in range(n):
    sys.stdout.write('\r')
    this_X, this_sp_labels = hp.make_features_sp(imgs[i],pca,canny_sigma,slic_compactness,slic_segments,hough_rel_thr,hough_max_lines,hough_canny,hough_radius,hough_threshold,hough_line_length,hough_line_gap,codebook)
    X.append(this_X)
    sp_labels.append(this_sp_labels)
    sys.stdout.write("%d/%d" % (i+1, n))
    sys.stdout.flush()
sys.stdout.write('\n')

print("Extracting ground-truths on " + str(n) + " images")
Y = list()
for i in range(n):
    sys.stdout.write('\r')
    this_y = np.asarray(hp.img_crop_sp(gt_imgs[i], sp_labels[i]))
    Y.append([])
    for j in range(len(this_y)):
        Y[-1].append(np.any(this_y[j]).astype(int))
    sys.stdout.write("%d/%d" % (i+1, n))
    sys.stdout.flush()
sys.stdout.write('\n')

print('Linearizing list of patches')
X_lin =  np.asarray([X[i][j] for i in range(len(X)) for j in range(len(X[i]))])
Y_lin =  np.asarray([Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))])

print('Rebalance classes by oversampling')
n_pos_to_add = (np.sum(Y_lin==0) - np.sum(Y_lin==1))
idx_to_duplicate = np.where(Y_lin==1)[0][np.random.randint(0,np.sum(Y_lin==1),n_pos_to_add)]
X_lin_aug = np.concatenate((X_lin,X_lin[idx_to_duplicate,:]),axis=0)
Y_lin_aug = np.concatenate((Y_lin,Y_lin[idx_to_duplicate]),axis=0)

#Print feature statistics
print('Computed ' + str(X_lin.shape[0]) + ' features')
print('Feature dimension = ' + str(X_lin.shape[1]))
print('Number of classes = ' + str(np.max(Y_lin)+1))

Y0 = [i for i, j in enumerate(Y_lin) if j == 0]
Y1 = [i for i, j in enumerate(Y_lin) if j == 1]
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
logreg = linear_model.LogisticRegression(C=7)

linreg = linear_model.LinearRegression()

#for the_classifier in {linreg,logreg}:
for the_classifier in {logreg}:
    print('Training classifier: ' + the_classifier.__class__.__name__)
    the_classifier.fit(X_lin, Y_lin)

    # Predict on the training set
    Z = the_classifier.predict(X_lin)
    thr_pred = 0.5

    # Get non-zeros in prediction and grountruth arrays
    Z_bin = np.zeros(Z.shape)
    Z_bin[np.where(Z>=thr_pred)[0]] = 1
    Z_bin[np.where(Z<thr_pred)[0]] = 0
    Z_bin = Z_bin.astype(float)

    f1_score = metrics.f1_score(Y_lin, Z_bin, average='weighted')
    conf_mat = metrics.confusion_matrix(Y_lin,Z_bin)
    TPR = conf_mat[0][0]/(conf_mat[0][0] + conf_mat[0][1])
    FPR = conf_mat[1][0]/(conf_mat[1][0] + conf_mat[1][1])
    print('Results with classifier: ' + the_classifier.__class__.__name__)
    print('TPR/FPR = ' + str(TPR) + '/' + str(FPR))
    print('F1-score = ' + str(f1_score))

print("Building CRF graphs")
X_crf = list()
Y_crf = list()
for i in range(n):
    sys.stdout.write('\r')
    vertices, edges = hp.make_graph_crf(sp_labels[i])
    edges_features = hp.make_edge_features(imgs[i],sp_labels[i],edges)
    X_crf.append((logreg.predict_proba(X[i]), np.asarray(edges), np.asarray(edges_features).reshape(-1,1)))
    sys.stdout.write("%d/%d" % (i+1, n))
    sys.stdout.flush()
sys.stdout.write('\n')

Y_flat = Y
Y_crf = [np.asarray(Y_flat[i]) for i in range(len(Y_flat))]

print("Training SSVM")
inference = 'qpbo'
# first, train on X with directions only:
crf = EdgeFeatureGraphCRF(inference_method=inference)
ssvm = OneSlackSSVM(crf, inference_cache=50, C=2., tol=.1, max_iter=500,
                    n_jobs=4)
ssvm.fit(X_crf, Y_crf)

print("Refining with SSVM")
Z_crf = ssvm.predict(X_crf)
Z_crf_lin =  np.asarray([Z_crf[i][j] for i in range(len(Z_crf)) for j in range(len(Z_crf[i]))])
Y_crf_lin = np.asarray([Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))])
f1_score = metrics.f1_score(Y_crf_lin, Z_crf_lin, average='weighted')
conf_mat = metrics.confusion_matrix(Y_crf_lin,Z_crf_lin)
TPR = conf_mat[0][0]/(conf_mat[0][0] + conf_mat[0][1])
FPR = conf_mat[1][0]/(conf_mat[1][0] + conf_mat[1][1])
print('Results with classifier: ' + ssvm.__class__.__name__)
print('TPR/FPR = ' + str(TPR) + '/' + str(FPR))
print('F1-score = ' + str(f1_score))

# Run prediction on the img_idx-th image
img_idx = 0
the_img = imgs[img_idx]
the_gt = gt_imgs[img_idx]

Xi, sp_labels_i = hp.make_features_sp(the_img,pca,canny_sigma,slic_compactness,slic_segments,hough_rel_thr,hough_max_lines,hough_canny,hough_radius,hough_threshold,hough_line_length,hough_line_gap,codebook)
Zi = logreg.predict_proba(Xi)
vertices, edges = hp.make_graph_crf(sp_labels[img_idx])
edges_features = hp.make_edge_features(the_img,sp_labels[img_idx],edges)
Xi_crf = (Zi, np.asarray(edges), np.asarray(edges_features).reshape(-1,1))
Zi_crf = np.asarray(ssvm.predict([Xi_crf])).ravel()

this_y = np.asarray(hp.img_crop_sp(the_gt, sp_labels_i))
Yi = list()
for j in range(len(this_y)):
    Yi.append(np.any(this_y[j]).astype(int))
Yi = np.asarray(Yi)

conf_mat = metrics.confusion_matrix(Yi,Zi_crf)
TPR = conf_mat[0][0]/(conf_mat[0][0] + conf_mat[0][1])
FPR = conf_mat[1][0]/(conf_mat[1][0] + conf_mat[1][1])
print('Results on image ' + str(img_idx) + ' with classifier: ' + ssvm.__class__.__name__)
print('TPR/FPR = ' + str(TPR) + '/' + str(FPR))
print('F1-score = ' + str(f1_score))

# Display prediction as an image
w = gt_imgs[img_idx].shape[0]
h = gt_imgs[img_idx].shape[1]

predicted_im = hp.sp_label_to_img(sp_labels_i, Zi_crf)
img_overlay = color.label2rgb(predicted_im,the_img,alpha=0.5)
img_over_conc = hp.concatenate_images(img_overlay,color.gray2rgb(gt_imgs[img_idx]))
predicted_labels = hp.concatenate_images(img_overlay,color.gray2rgb(predicted_im))
fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size
plt.imshow(img_over_conc, cmap='Greys_r');
plt.title('im ' + str(img_idx) + ', ' +  ssvm.__class__.__name__ + ' TPR/FPR = ' + str(TPR) + '/' + str(FPR))
plt.show()
