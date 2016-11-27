import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import helpers as hp
from PIL import Image
from skimage import (morphology, feature,color)
from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
import skimage.draw

# Loaded a set of images
root_dir = "training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = min(20, len(files)) # Load maximum 20 images
print("Loading " + str(n) + " images")
imgs = [hp.load_image(image_dir + files[i]) for i in range(n)]
print(files[0])

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [hp.load_image(gt_dir + files[i]) for i in range(n)]
print(files[0])

n = 10 # Only use 10 images for training

print('Image size = ' + str(imgs[0].shape[0]) + ',' + str(imgs[0].shape[1]))

# Show first image and its groundtruth image
cimg = hp.concatenate_images(imgs[0], gt_imgs[0])
fig1 = plt.figure(figsize=(10, 10))
plt.imshow(cimg, cmap='Greys_r'); plt.show()

hough_lines = list()
for i in range(n):
    im = color.rgb2gray(imgs[i])
    hough_lines.append(np.zeros((im.shape[0],im.shape[1])))
    edge = feature.canny(im)
    lines = probabilistic_hough_line(edge, threshold=10, line_length=45,line_gap=3)
    for line in lines:
        p0, p1 = line
        line_idx = skimage.draw.line(p0[1], p0[0], p1[1], p1[0])
        hough_lines[-1][line_idx] = 1


# Extract patches from input images
patch_size = 16 # each patch is 16*16 pixels

hough_patches = [hp.img_crop(hough_lines[i], patch_size, patch_size) for i in range(n)]
img_patches = [hp.img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
gt_patches = [hp.img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

test = img_patches[0][0]
desc0 = feature.daisy(color.rgb2gray(img_patches[0][0]),step=2,radius=7)

# Linearize list of patches
hough_patches = np.asarray([hough_patches[i][j] for i in range(len(hough_patches)) for j in range(len(hough_patches[i]))])
img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

# Compute features for each image patch
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

X_mean_var = np.asarray([ hp.extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
#X_daisy = np.asarray([ feature.daisy(color.rgb2gray(img_patches[i]),step=2,radius=7)[0,0,:] for i in range(len(img_patches))])
X_d0 = np.asarray([ feature.daisy(img_patches[i][:,:,0],step=2,radius=7)[0,0,:] for i in range(len(img_patches))])
X_d1 = np.asarray([ feature.daisy(img_patches[i][:,:,1],step=2,radius=7)[0,0,:] for i in range(len(img_patches))])
X_d2 = np.asarray([ feature.daisy(img_patches[i][:,:,2],step=2,radius=7)[0,0,:] for i in range(len(img_patches))])
X = np.concatenate((X_mean_var,X_d0,X_d1,X_d2),axis=1)
Y = np.asarray([hp.value_to_class(np.mean(gt_patches[i]),foreground_threshold) for i in range(len(gt_patches))])

# Print feature statistics
print('Computed ' + str(X.shape[0]) + ' features')
print('Feature dimension = ' + str(X.shape[1]))
print('Number of classes = ' + str(np.max(Y)))

Y0 = [i for i, j in enumerate(Y) if j == 0]
Y1 = [i for i, j in enumerate(Y) if j == 1]
print('Class 0: ' + str(len(Y0)) + ' samples')
print('Class 1: ' + str(len(Y1)) + ' samples')

# Display a patch that belongs to the foreground class
plt.imshow(gt_patches[Y1[130]], cmap='Greys_r');
plt.title('Example of positive patch')
plt.show()

# Plot 2d features using groundtruth to color the datapoints
plt.scatter(X_daisy[:, 10], X_daisy[:, 20], c=Y, edgecolors='k', cmap=plt.cm.Paired); plt.show()
plt.scatter(X_mean_var[:, 0], X_mean_var[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired); plt.show()

# train a logistic regression classifier
from sklearn import linear_model

# we create an instance of the classifier and fit the data
logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
logreg.fit(X, Y)

#linreg = linear_model.LinearRegression()
linreg = linear_model.Ridge(alpha=1.0)
linreg.fit(X, Y)

# Predict on the training set
Z = logreg.predict_proba(X)[:,1]
#Z = linreg.predict(X)

thr_pred = 0.5

# Get non-zeros in prediction and grountruth arrays
Z_pos = np.where(Z>=thr_pred)[0]
Z_neg = np.where(Z<thr_pred)[0]
Y_pos = np.nonzero(Y)[0]
Y_neg = np.where(Y==0)[0]

TPR = len(list(set(Y_pos) & set(Z_pos))) / float(len(Z))
FPR = len(list(set(Y_neg) & set(Z_pos))) / float(len(Z))
print('TPR/FPR = ' + str(TPR) + '/' + str(FPR))
TPR = len(list(set(Yn) & set(Zn))) / float(len(Z))
