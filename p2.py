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

idx = 8
print("Fitting k-means on images")
img_sample = utils.shuffle(imgs[0].reshape(imgs[0].shape[0]*imgs[0].shape[1],-1), random_state=0)[:1000]
codebook_colors,_ = cluster.vq.kmeans(img_sample,n_colors,thresh=1)
codes, _ = cluster.vq.vq(imgs[idx].reshape(imgs[idx].shape[0]*imgs[idx].shape[1],-1), codebook_colors)
img_vq = hp.recreate_image(codebook_colors,codes,imgs[idx].shape[0],imgs[idx].shape[1])
plt.imshow(hp.concatenate_images(img_vq,imgs[idx]));
plt.title('vector quantization with ' + str(n_colors) + ' colors');
plt.show()

the_img = imgs[idx]
the_gt = gt_imgs[idx]

img_and_gt = color.label2rgb(the_gt.astype(bool),the_img)
hough = hp.make_hough(the_img,0.5,25,sig_canny=1,radius=20,threshold=1,line_length=200,line_gap=4)
hough_img = hp.concatenate_images(img_and_gt,color.rgb2gray(hough))
plt.imshow(hough_img); plt.title('thresholding, hough transform, binary dilation'); plt.show()

#features are extracted from n_im_pca images on a dense grid (grid_step). PCA is then applied for dimensionality reduction.
print('Extracting edges')
X_dt = hp.get_features_edges(imgs[0:n],grid_step,canny_sigma)
print('Extracting euclidean distance transform')
X_dt = hp.get_features_dt(imgs[0:n],canny_sigma,grid_step)
print('Extracting mean RGB values on vector quantized patches')
X_vq = hp.get_features_vq_colors(imgs[0:n],grid_step,codebook_colors)
print('Extracting hough transforms')
X_hough = hp.get_features_hough(imgs[0:n],0.4,20,grid_step,canny_sigma,radius=1,threshold=10,line_length=45,line_gap=3)
print('Extracting SIFT features')
X_sift = hp.get_features_sift(imgs[0:n],canny_sigma,sift_sigmas,grid_step)
print('Fitting PCA model on SIFT with n_components = ' + str(n_components_pca))
pca = decomposition.PCA(n_components_pca)
pca.fit(X_sift)
X_sift = pca.transform(X_sift)
X = np.concatenate((X_sift, X_hough,X_vq,X_dt),axis=1)

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

print('Training Adaboost classifier')
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=500)
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

my_classifier = bdt
# Run prediction on the img_idx-th image
img_idx = 0
the_img = imgs[img_idx]
the_gt = gt_imgs[img_idx]

X_dt = hp.get_features_dt([imgs[img_idx]],canny_sigma,grid_step)
X_vq = hp.get_features_vq_colors([imgs[img_idx]],grid_step,codebook_colors)
X_hough = hp.get_features_hough([imgs[img_idx]],0.4,20,grid_step,canny_sigma,radius=1,threshold=10,line_length=45,line_gap=3)
X_sift = hp.get_features_sift([imgs[img_idx]],canny_sigma,sift_sigmas,grid_step)
pca = decomposition.PCA(n_components_pca)
pca.fit(X_sift)
X_sift = pca.transform(X_sift)
Xi = np.concatenate((X_sift, X_hough,X_vq,X_dt),axis=1)

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
