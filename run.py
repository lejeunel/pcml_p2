import re
from sklearn import (cluster,decomposition,preprocessing,utils,metrics)
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import helpers as hp
from PIL import Image
from skimage import (morphology, feature,color,transform,filters,draw,transform,data, segmentation, color)
from scipy import cluster
from skimage.future import graph
from matplotlib import pyplot as plt
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import OneSlackSSVM
import ssvm
from sklearn.model_selection import (GridSearchCV,KFold,cross_val_score)
from sklearn import (linear_model,preprocessing)
from sklearn.ensemble import RandomForestClassifier
import mask_to_submission as submit

#Parameters
n = 94 #Num of images for training
grid_step = 16 #keypoints are extracted every grid_step pixels
n_im_pca = 10 #Num of images for PCA
n_components_pca = 60 #Num of components for PCA compression
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
slic_segments = 600
slic_compactness = 30
cv_folds = 5

root_dir = "training/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
files = sorted(files, key=lambda x: (int(re.sub('\D','',x)),x))

#Removing obvious outliers
outliers = np.array([10,20,26,27,64,76])
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

#Visual check of hough transform parameters
idx = 11
the_img = imgs[idx]
the_gt = gt_imgs[idx]
img_and_gt = color.label2rgb(the_gt.astype(bool),the_img)
hough = hp.make_hough(the_img,0.5,25,sig_canny=1,radius=20,threshold=1,line_length=100,line_gap=3)
hough_img = hp.concatenate_images(img_and_gt,color.rgb2gray(hough))
plt.imshow(hough_img); plt.title('thresholding, hough transform, binary dilation');
plt.savefig('ex_hough.eps')
#plt.show()

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
plt.autoscale(False)
# overlay graph:
for edge in edges:
    plt.plot([centers[edge[0]][0],centers[edge[1]][0]],
             [centers[edge[0]][1],centers[edge[1]][1]],'w',alpha=0.3)
plt.savefig('ex_graph.png')
#plt.show()

#features are extracted from n_im_pca images on a dense grid (grid_step). PCA is then fit for dimensionality reduction.
print('Extracting SIFT features')
X_sift = hp.get_features_sift(imgs[0:n_im_pca],canny_sigma,sift_sigmas,grid_step)
print('Fitting PCA model on SIFT with n_components = ' + str(n_components_pca))
pca = decomposition.PCA(n_components_pca)
pca.fit(X_sift)
X_sift = pca.transform(X_sift)

plt.figure()
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.title('PCA  compression of SIFT descriptors')
plt.savefig('ex_pca_explained_variance.eps')
#plt.show()

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
        Y[-1].append((np.sum(this_y[j])/this_y[j].shape[0] > foreground_threshold).astype(int))
    sys.stdout.write("%d/%d" % (i+1, n))
    sys.stdout.flush()
sys.stdout.write('\n')

print('Linearizing list of patches')
X_lin =  np.asarray([X[i][j] for i in range(len(X)) for j in range(len(X[i]))])
Y_lin =  np.asarray([Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))])

#Print feature statistics
print('Computed ' + str(X_lin.shape[0]) + ' features')
print('Feature dimension = ' + str(X_lin.shape[1]))
print('Number of classes = ' + str(np.max(Y_lin)+1))

Y0 = [i for i, j in enumerate(Y_lin) if j == 0]
Y1 = [i for i, j in enumerate(Y_lin) if j == 1]
print('Class 0: ' + str(len(Y0)) + ' samples')
print('Class 1: ' + str(len(Y1)) + ' samples')

print("Building CRF graphs")
X_crf = list()
Y_crf = list()
for i in range(n):
    sys.stdout.write('\r')
    vertices, edges = hp.make_graph_crf(sp_labels[i])
    edges_features = hp.make_edge_features(imgs[i],sp_labels[i],edges)
    X_crf.append((X[i], np.asarray(edges), np.asarray(edges_features).reshape(-1,1)))
    sys.stdout.write("%d/%d" % (i+1, n))
    sys.stdout.flush()
sys.stdout.write('\n')

Y_crf = [np.asarray(Y[i]) for i in range(len(Y))]


gen_estim = linear_model.LogisticRegression()
print('cross-validation on generic estimator')
X_gen =  np.asarray([X_crf[i][0][j] for i in range(len(X_crf)) for j in range(len(X_crf[i][0]))])
Y_gen =  np.asarray([Y_crf[i][j] for i in range(len(Y_crf)) for j in range(len(Y_crf[i]))])
scorer = metrics.make_scorer(metrics.f1_score)
scores_pre = list()
C_pre = np.logspace(-3,7,40)
for this_C in C_pre:
    gen_estim.set_params(C = this_C)
    scores_pre.append(cross_val_score(gen_estim, X_gen, Y_gen, cv=cv_folds,scoring=scorer))
    print('F-1 score (C, mean +/- 2*std) = ' + str(this_C) + ', ' + str(np.mean(scores_pre[-1])) + '/' + str(np.std(scores_pre[-1]) ))
scores_pre = np.asarray(scores_pre)

print('cross-validation on SSVM estimator')
gen_estim = linear_model.LogisticRegression(C = 10e-2)
scores_crf = list()
C_crf = np.logspace(-3,5,15)
for this_C in C_crf:
    my_ssvm = ssvm.MySSVM(gen_estim,C_ssvm=this_C,inference='qpbo',n_jobs=1)
    scores_crf.append(cross_val_score(my_ssvm, X_crf, Y_crf, cv=cv_folds))
    print('F-1 score (C, mean +/- 2*std) = ' + str(this_C) + ', ' + str(np.mean(scores_crf[-1])) + '/' + str(np.std(scores_crf[-1]) ))
scores_crf = np.asarray(scores_crf)

print('cross-validation on SSVM estimator with Random Forest')
gen_estim = RandomForestClassifier(n_estimators=50,max_depth=10)
scores_crf_rf = list()
C_crf_rf = np.logspace(-3,5,15)
for this_C in C_crf_rf:
    my_ssvm = ssvm.MySSVM(gen_estim,C_ssvm=this_C,inference='qpbo',n_jobs=1)
    scores_crf_rf.append(cross_val_score(my_ssvm, X_crf, Y_crf, cv=cv_folds))
    print('F-1 score (C, mean +/- 2*std) = ' + str(this_C) + ', ' + str(np.mean(scores_crf_rf[-1])) + '/' + str(np.std(scores_crf_rf[-1]) ))
scores_crf_rf = np.asarray(scores_crf_rf)

plt.semilogx(C_pre,np.mean(scores_pre,axis=1),'b',label='LogReg')
plt.semilogx(C_pre,np.mean(scores_pre,axis=1) + np.std(scores_pre,axis=1),'b--')
plt.semilogx(C_pre,np.mean(scores_pre,axis=1) - np.std(scores_pre,axis=1),'b--')
plt.semilogx(C_crf,np.mean(scores_crf,axis=1),'r',label='LogReg + SSVM')
plt.semilogx(C_crf,np.mean(scores_crf,axis=1) + np.std(scores_crf_rf,axis=1),'r--')
plt.semilogx(C_crf,np.mean(scores_crf,axis=1) - np.std(scores_crf_rf,axis=1),'r--')
plt.semilogx(C_crf_rf,np.mean(scores_crf_rf,axis=1),'g',label='Random Forest + SSVM')
plt.semilogx(C_crf_rf,np.mean(scores_crf_rf,axis=1) + np.std(scores_crf_rf,axis=1),'g--')
plt.semilogx(C_crf_rf,np.mean(scores_crf_rf,axis=1) - np.std(scores_crf_rf,axis=1),'g--')
plt.xlabel('regularization factor (log10)')
plt.ylabel('F1-score')
plt.title(str(cv_folds) + '-fold cross validation')
plt.legend(loc='bottom right')
plt.savefig('cross_val_ssvm.png')
plt.show()

# Run prediction on the img_idx-th image
img_idx = 1
the_img = imgs[img_idx]
the_gt = gt_imgs[img_idx]

#Train with everything except img_idx img
X_gen_train =  np.asarray([X_crf[i][0][j] for i in range(len(X_crf)) for j in range(len(X_crf[i][0])) if(i!= img_idx)])
Y_gen_train =  np.asarray([Y_crf[i][j] for i in range(len(Y_crf)) for j in range(len(Y_crf[i])) if(i != img_idx)])
X_crf_train = [X_crf[i] for i in range(len(X_crf)) if(i != img_idx)]
Y_crf_train = [Y_crf[i] for i in range(len(Y_crf)) if(i != img_idx)]

gen_estim_rf = RandomForestClassifier(n_estimators=50,max_depth=10)
gen_estim_logreg = linear_model.LogisticRegression(C = 10e-2)
my_ssvm = ssvm.MySSVM(gen_estim_rf,C_ssvm=1,n_jobs=1)
print('Fitting all models')
my_ssvm.fit(X_crf,Y_crf)
gen_estim_rf.fit(X_gen_train,Y_gen_train)
gen_estim_logreg.fit(X_gen_train,Y_gen_train)

print('Making features on unseen image: ' + str(img_idx))
Xi_gen, sp_labels_i = hp.make_features_sp(the_img,pca,canny_sigma,slic_compactness,slic_segments,hough_rel_thr,hough_max_lines,hough_canny,hough_radius,hough_threshold,hough_line_length,hough_line_gap,codebook)
vertices, edges = hp.make_graph_crf(sp_labels[img_idx])
edges_features = hp.make_edge_features(the_img,sp_labels[img_idx],edges)

print('Predicting all models on unseen image: ' + str(img_idx))
Xi_crf = [(Xi_gen, np.asarray(edges), np.asarray(edges_features).reshape(-1,1))]
Zi_crf = np.asarray(my_ssvm.predict(Xi_crf)).ravel()
Zi_gen_logreg = np.asarray(gen_estim_logreg.predict(Xi_gen)).ravel()
Zi_gen_rf = np.asarray(gen_estim_rf.predict(Xi_gen)).ravel()
f1_score_ssvm = metrics.f1_score(Y_crf[img_idx],Zi_crf)
f1_score_logreg = metrics.f1_score(Y_crf[img_idx],Zi_gen_logreg)
f1_score_rf = metrics.f1_score(Y_crf[img_idx],Zi_gen_rf)

this_y = np.asarray(hp.img_crop_sp(the_gt, sp_labels_i))
Yi = list()
for j in range(len(this_y)):
    Yi.append(np.any(this_y[j]).astype(int))
Yi = np.asarray(Yi)

print('Results on image ' + str(img_idx) + ' with logreg ')
print('F1-score = ' + str(f1_score_logreg))
print('Results on image ' + str(img_idx) + ' with RF ')
print('F1-score = ' + str(f1_score_rf))
print('Results on image ' + str(img_idx) + ' with refined RF')
print('F1-score = ' + str(f1_score_ssvm))

# Display prediction as an image
w = gt_imgs[img_idx].shape[0]
h = gt_imgs[img_idx].shape[1]

predicted_im_logreg = hp.sp_label_to_img(sp_labels_i, Zi_gen_logreg)
predicted_im_rf = hp.sp_label_to_img(sp_labels_i, Zi_gen_rf)
predicted_im_ssvm = hp.sp_label_to_img(sp_labels_i, Zi_crf)
img_overlay_logreg = color.label2rgb(predicted_im_logreg,the_img,alpha=0.5)
img_overlay_rf = color.label2rgb(predicted_im_rf,the_img,alpha=0.5)
img_overlay_ssvm = color.label2rgb(predicted_im_ssvm,the_img,alpha=0.5)
plt.subplot(221) # create a figure with the default size
plt.imshow(img_overlay_logreg);
plt.axis('off')
plt.subplot(222) # create a figure with the default size
plt.imshow(img_overlay_rf);
plt.axis('off')
plt.subplot(223) # create a figure with the default size
plt.imshow(img_overlay_ssvm);
plt.axis('off')
plt.subplot(224) # create a figure with the default size
plt.imshow(1-the_gt)
plt.axis('off')
#plt.title('im ' + str(img_idx) + ', ' +  the_classifier.__class__.__name__ + ' TPR/FPR = ' + str(TPR) + '/' + str(FPR))
plt.savefig('prediction.png')
plt.show()

#get test set
files_test = list()
imgs_test = list()
test_dir = 'test_set_images'
test_sub_dirs = os.listdir(test_dir)
for i in range(len(test_sub_dirs)):
    this_dir = os.path.join(os.path.join(test_dir,test_sub_dirs[i]))
    files_test.append(os.path.join(this_dir,os.listdir(this_dir)[0]))

test_imgs = [hp.load_image(files_test[i]) for i in range(len(files_test))]

#Fit and predict test set
my_ssvm = ssvm.MySSVM(gen_estim,C_ssvm=1900,n_jobs=1)
print('Fitting using whole training set')
my_ssvm.fit(X_crf,Y_crf)

print('Extracting features on ' + str(len(test_imgs))+ ' test images')
X_test = list()
sp_labels_test = list()
for i in range(len(test_imgs)):
    sys.stdout.write('\r')
    this_X, this_sp_labels = hp.make_features_sp(test_imgs[i],pca,canny_sigma,slic_compactness,slic_segments,hough_rel_thr,hough_max_lines,hough_canny,hough_radius,hough_threshold,hough_line_length,hough_line_gap,codebook)
    X_test.append(this_X)
    sp_labels_test.append(this_sp_labels)
    sys.stdout.write("%d/%d" % (i+1, len(test_imgs)))
    sys.stdout.flush()
sys.stdout.write('\n')

print('Building CRF graphs on ' + str(len(test_imgs))+ ' test images')
X_crf_test = list()
for i in range(len(test_imgs)):
    sys.stdout.write('\r')
    vertices, edges = hp.make_graph_crf(sp_labels_test[i])
    edges_features = hp.make_edge_features(test_imgs[i],sp_labels_test[i],edges)
    X_crf_test.append((X_test[i], np.asarray(edges), np.asarray(edges_features).reshape(-1,1)))
    sys.stdout.write("%d/%d" % (i+1, len(test_imgs)))
    sys.stdout.flush()
sys.stdout.write('\n')

print('Predicting on test set')
Z_crf_test = my_ssvm.predict(X_crf_test)

test_dir_res= 'test_set_res'
print('Writing segmentations on test set')
#im = hp.sp_label_to_img(sp_labels_test[0],Z_crf_test[0])
#im_overlay = color.label2rgb(im,test_imgs[0],alpha=0.5)
#plt.imshow(im_overlay); plt.show()
assert os.path.isdir(test_dir_res), "Directory " + test_dir_res + " must exist!"

res_paths = list()
for i in range(len(Z_crf_test)):
    this_out_filename = os.path.splitext(os.path.basename(files_test[i]))[0]
    this_im = hp.sp_label_to_img(sp_labels_test[i],Z_crf_test[i])
    this_im_arr = Image.fromarray((this_im * 255).astype(np.uint8))
    this_path = os.path.join(test_dir_res,this_out_filename + '.png')
    this_im_arr.save(this_path)
    res_paths.append(this_path)

print('Writing submission file')
submit.masks_to_submission('submission.csv',res_paths)
print('Done')
