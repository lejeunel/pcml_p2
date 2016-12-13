import re
from sklearn import (cluster,decomposition,preprocessing,utils,metrics)
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import helpers as hp
from PIL import Image
from skimage import (morphology, feature,color,transform,filters,segmentation)
from scipy import cluster
#from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
from skimage import (draw,transform)
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import OneSlackSSVM

#Parameters
n = 90 #Num of images for training
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
slic_segments = 400

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

idx = 0
labels = segmentation.slic(imgs[idx], compactness=30, n_segments=slic_segments)
vertices, edges = hp.make_graph_crf(labels)
# compute region centers:
gridx, gridy = np.mgrid[:imgs[0].shape[0], :imgs[0].shape[1]]
centers = dict()
for v in vertices:
    centers[v] = [gridy[labels == v].mean(), gridx[labels == v].mean()]

# plot labels
plt.imshow(color.label2rgb(labels,imgs[idx],kind='avg'))
# overlay graph:
for edge in edges:
    plt.plot([centers[edge[0]][0],centers[edge[1]][0]],
             [centers[edge[0]][1],centers[edge[1]][1]])
plt.show()

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " ground-truth images")
#gt_imgs = [hp.load_image(gt_dir + files[i]) for i in range(n)]
gt_imgs = [hp.load_image(gt_dir + files[i]) for i in range(len(files))]

print('Image size = ' + str(imgs[0].shape[0]) + ',' + str(imgs[0].shape[1]))

from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt

idx = 11

the_img = imgs[idx]
the_gt = gt_imgs[idx]

n_im_kmeans = 5
#features are extracted from n_im_pca images on a dense grid (grid_step). PCA is then applied for dimensionality reduction.
print('Extracting RGB histograms')
slic_imgs = [color.label2rgb(segmentation.slic(imgs[i], compactness=30, n_segments=slic_segments), imgs[i], kind='avg') for i in range(n_im_kmeans)]
X_rgb = hp.get_features_rgb(slic_imgs[0:n_im_kmeans],grid_step)
print('Extracting euclidean distance transform')
X_dt = hp.get_features_dt(imgs[0:n_im_kmeans],canny_sigma,grid_step)
print('Extracting SIFT features')
X_sift = hp.get_features_sift(imgs[0:n_im_kmeans],canny_sigma,sift_sigmas,grid_step)

print('Fitting PCA model on SIFT with n_components = ' + str(n_components_pca))
pca = decomposition.PCA(n_components_pca)
pca.fit(X_sift)
X_sift = pca.transform(X_sift)
X = np.concatenate((X_sift, X_rgb,X_dt),axis=1)

print('Generating codebook on ' + str(X.shape[0]) + ' samples  with ' + str(n_clusters_bow) + ' clusters')
codebook, distortion = cluster.vq.kmeans(X, n_clusters_bow,thresh=1)

codes = list()
features = list()
X = list()
print('Generating codes on ' + str(n) + ' images')
for i in range(n):
    sys.stdout.write('\r')
    this_labels = segmentation.slic(imgs[i], compactness=30, n_segments=slic_segments)
    this_rgb = color.label2rgb(this_labels, imgs[i], kind='avg').reshape(-1,3)
    this_sift = hp.get_features_sift([imgs[i]],canny_sigma,sift_sigmas,1)
    this_sift = pca.transform(this_sift)
    this_dt = hp.get_features_dt([imgs[i]],canny_sigma,1)
    this_feats = np.concatenate((this_sift, this_rgb,this_dt),axis=1)
    code, dist = cluster.vq.vq(this_feats, codebook)
    code = code.reshape(int(imgs[0].shape[0]),int(imgs[0].shape[1]))
    patch_codes = hp.img_crop_sp(code, this_labels)
    hist_codes = np.asarray([np.histogram(patch_codes[i], n_clusters_bow)[0] for i in range(len(patch_codes)) ])
    features = hist_codes
    vertices, edges = hp.make_graph_crf(this_labels)
    edges_features = hp.make_edge_features(imgs[i],this_labels,edges)
    X.append((features, np.asarray(edges), np.asarray(edges_features).reshape(-1,1)))
    sys.stdout.write("%d/%d" % (i+1, n))
    sys.stdout.flush()
    codes.append(code)
sys.stdout.write('\n')

#gt_patches = [hp.img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

# Compute features for each image patch
Y = list()
print('Building ground-truth array')
for i in range(n):
    labels = segmentation.slic(imgs[i], compactness=30, n_segments=slic_segments)
    this_y = list()
    for j in np.unique(labels):
        this_sp_mask = np.where(labels == j)
        this_gt_sp = gt_imgs[i][this_sp_mask[0],this_sp_mask[1]]
        #this_y = np.asarray([hp.value_to_class(this_gt_sp,foreground_threshold) for j in range(len(gt_patches[i]))]).reshape(int(gt_imgs[0].shape[0]/grid_step),int(gt_imgs[0].shape[0]/grid_step)).ravel()
        this_y.append(hp.value_to_class(this_gt_sp,foreground_threshold))
    Y.append(np.asarray(this_y))


# train a logistic regression classifier
from sklearn import (linear_model,svm,preprocessing)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

print("Training SSVM")
inference = 'qpbo'
# first, train on X with directions only:
crf = EdgeFeatureGraphCRF(inference_method=inference)
ssvm = OneSlackSSVM(crf, inference_cache=50, C=1., tol=.1, max_iter=500,
                    n_jobs=4)

Y_flat = [y_.ravel() for y_ in Y]
Y_flat = np.asarray([Y_flat[i][j] for i in range(len(Y_flat)) for j in range(len(Y_flat[i]))])
ssvm.fit(X, Y)
Z_bin = ssvm.predict(X)
Z_flat = np.asarray([Z_bin[i][j] for i in range(len(Z_bin)) for j in range(len(Z_bin[i]))])
f1_score = metrics.f1_score(Y_flat, Z_flat, average='weighted')
conf_mat = metrics.confusion_matrix(Y_flat,Z_flat)
TPR = conf_mat[0][0]/(conf_mat[0][0] + conf_mat[0][1])
FPR = conf_mat[1][0]/(conf_mat[1][0] + conf_mat[1][1])
print('Results with classifier: ' + ssvm.__class__.__name__)
print('TPR/FPR = ' + str(TPR) + '/' + str(FPR))
print('F1-score = ' + str(f1_score))

my_classifier = ssvm
# Run prediction on the img_idx-th image
img_idx = 0
the_img = imgs[img_idx]
the_gt = gt_imgs[img_idx]

Xi = list()
this_labels = segmentation.slic(the_img, compactness=30, n_segments=slic_segments)
this_rgb = color.label2rgb(this_labels, the_img, kind='avg').reshape(-1,3)
this_sift = hp.get_features_sift([the_img],canny_sigma,sift_sigmas,1)
this_sift = pca.transform(this_sift)
this_dt = hp.get_features_dt([the_img],canny_sigma,1)
this_feats = np.concatenate((this_sift, this_rgb,this_dt),axis=1)
code, dist = cluster.vq.vq(this_feats, codebook)
code = code.reshape(int(imgs[0].shape[0]),int(imgs[0].shape[1]))
patch_codes = hp.img_crop_sp(code, this_labels)
hist_codes = np.asarray([np.histogram(patch_codes[i], n_clusters_bow)[0] for i in range(len(patch_codes)) ])
features = hist_codes
vertices, edges = hp.make_graph_crf(this_labels)
edges_features = hp.make_edge_features(imgs[i],this_labels,edges)
Xi.append((features, np.asarray(edges), np.asarray(edges_features).reshape(-1,1)))

Zi = my_classifier.predict(Xi)
Zi_flat = np.asarray([Zi[i] for i in range(len(Zi))])

# Compute features for each image patch
Yi = list()
print('Building ground-truth array')
labels = segmentation.slic(the_img, compactness=30, n_segments=slic_segments)
this_y = list()
for j in np.unique(labels):
    this_sp_mask = np.where(labels == j)
    this_gt_sp = the_gt[this_sp_mask[0],this_sp_mask[1]]
    this_y.append(hp.value_to_class(this_gt_sp,foreground_threshold))
Yi.append(np.asarray(this_y))

Yi_flat = np.asarray([y_.ravel() for y_ in Yi])
conf_mat = metrics.confusion_matrix(Yi_flat.ravel(),Zi_flat[0].ravel())
TPR = conf_mat[0][0]/(conf_mat[0][0] + conf_mat[0][1])
FPR = conf_mat[1][0]/(conf_mat[1][0] + conf_mat[1][1])
print('TPR/FPR = ' + str(TPR) + '/' + str(FPR))

# Display prediction as an image
w = gt_imgs[img_idx].shape[0]
h = gt_imgs[img_idx].shape[1]
predicted_im = hp.sp_label_to_img(labels, Zi[0])

img_overlay = color.label2rgb(predicted_im,the_img,alpha=0.5)
img_over_conc = hp.concatenate_images(img_overlay,color.gray2rgb(gt_imgs[img_idx]))
predicted_labels = hp.concatenate_images(img_overlay,color.gray2rgb(predicted_im))
fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size
plt.imshow(img_over_conc, cmap='Greys_r');
plt.title('im ' + str(img_idx) + ', ' +  my_classifier.__class__.__name__ + 'TPR/FPR = ' + str(TPR) + '/' + str(FPR))
plt.show()
