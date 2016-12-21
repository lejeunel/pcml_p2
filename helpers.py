import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import sys
sys.path.append('/usr/lib/python3.5/site-packages')
import cv2
from skimage import (color,feature,filters,draw,morphology,segmentation)
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line,rescale)
from scipy import cluster
from skimage.filters import gabor_kernel
from skimage.future import graph
from pystruct.utils import make_grid_edges, edge_list_to_features

def make_edge_features(img,labels,edges):
    """
    Computes probabilities associated to edges of CRF graphs.
    They are calculated as L(i,j)/(1+ || si - sj||) where:
    L(i,j) is the length (in pixels) of the boundary between superpixel i and j,
    si and sj are respectively the mean colors of superpixels i and j
    """

    img = color.rgb2luv(img)
    edge_features = list()
    selem = morphology.square(2)
    for i in range(len(edges)):
        mask1 = labels == edges[i][0]
        mask2 = labels == edges[i][1]
        mask1_idx = np.where(mask1)
        mask2_idx = np.where(mask2)
        mask1_dilate = morphology.binary_dilation(mask1,selem=selem)
        mask2_dilate = morphology.binary_dilation(mask2,selem=selem)
        inter_bounds = mask1_dilate*mask2_dilate
        this_pix1 = img[mask1_idx[0],mask1_idx[1],:]
        this_pix2 = img[mask2_idx[0],mask2_idx[1],:]
        denom = 1+np.linalg.norm(np.mean(this_pix1,axis=0)-np.mean(this_pix2,axis=0))
        numer = np.sum(inter_bounds)
        edge_features.append(numer/(denom))

    return edge_features

def make_graph_crf(labels):
    """
    Makes the vertices and edges.
    The index of the vertices are given by the values of labels (2D array)
    Edges connect neighboring labels
    """

    # get unique labels
    vertices = np.unique(labels)

    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices,np.arange(len(vertices))))
    labels = np.array([reverse_dict[x] for x in labels.flat]).reshape(labels.shape)

    # create edges
    down = np.c_[labels[:-1, :].ravel(), labels[1:, :].ravel()]
    right = np.c_[labels[:, :-1].ravel(), labels[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges,axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:,0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[int(x%num_vertices)],
              vertices[int(x/num_vertices)]] for x in edges]

    return vertices, edges

def get_features_edges(imgs,grid_step,canny_sigma):
    """
    Computes canny edge features (not used9
    """

    w = imgs[0].shape[1]
    h = imgs[0].shape[0]
    X_edge = np.asarray([ feature.canny(color.rgb2gray(imgs[i]),sigma=canny_sigma) for i in range(len(imgs))])
    patches = [img_crop(X_edge[i], grid_step, grid_step) for i in range(len(X_edge))]
    patches = np.asarray([np.mean(patches[i][j]).astype(int) for i in range(len(patches)) for j in range(len(patches[i]))])

    return patches.reshape(-1,1)

def get_features_rgb(imgs,grid_step=16,labels=None):

    w = imgs[0].shape[1]
    h = imgs[0].shape[0]
    if(labels is not None):
        X_rgb = [ np.mean(imgs[i][np.where(labels[i] == j)[0],np.where(labels[i] == j)[1],:],axis=0) for i in range(len(imgs)) for j in np.unique(labels[i])]
        return X_rgb

    else:
        patches = [img_crop(imgs[i], grid_step, grid_step) for i in range(len(imgs))]
        patches = np.asarray([np.mean(patches[i][j].reshape(grid_step**2,-1),axis=0) for i in range(len(patches)) for j in range(len(patches[i]))])

        return patches.reshape(-1,3)

def get_features_dt(imgs,canny_sigma,grid_step=16,labels=None):

    w = imgs[0].shape[1]
    h = imgs[0].shape[0]
    if(labels is not None):
        X_dt = np.asarray([ distance_transform_edge(color.rgb2gray(imgs[i]),edge_sigma=canny_sigma) for i in range(len(imgs))])
        X_dt = [ np.mean(X_dt[i][np.where(labels[i] == j)[0],np.where(labels[i] == j)[1]],axis=0) for i in range(len(imgs)) for j in np.unique(labels[i])]
        return X_dt

    else:
        X_dt = np.asarray([ distance_transform_edge(color.rgb2gray(imgs[i]),edge_sigma=canny_sigma) for i in range(len(imgs))])
        patches = [img_crop(X_dt[i], grid_step, grid_step) for i in range(len(X_dt))]
        patches = np.asarray([np.mean(patches[i][j]).astype(int) for i in range(len(patches)) for j in range(len(patches[i]))])

    return patches.reshape(-1,1)

def get_features_vq_colors(imgs,grid_step,codebook):

    w = imgs[0].shape[1]
    h = imgs[0].shape[0]
    X_rgb = [ recreate_image(codebook,cluster.vq.vq(imgs[i].reshape(w*h,-1), codebook)[0],w,h) for i in range(len(imgs))]
    patches = [img_crop(X_rgb[i], grid_step, grid_step) for i in range(len(X_rgb))]
    patches = np.asarray([np.mean(patches[i][j]).astype(int) for i in range(len(patches)) for j in range(len(patches[i]))])

    return patches.reshape(-1,1)

def get_features_hist(imgs,n_bins,grid_step):

    patches = [img_crop(imgs[i], grid_step, grid_step) for i in range(len(imgs))]
    patches = np.asarray([np.asarray((np.histogram(patches[i][j][:,:,0],n_bins)[0],
                            np.histogram(patches[i][j][:,:,1],n_bins)[0],
                            np.histogram(patches[i][j][:,:,2],n_bins)[0])).reshape(1,-1)
                           for i in range(len(patches)) for j in range(len(patches[i]))])

    return patches.reshape(-1,3*n_bins)


def distance_transform_edge(img,edge_sigma=1):
    """
    Computes the taxicab distance transform on binary array.
    """

    edge_map = feature.canny(img,sigma=edge_sigma)
    dt = ndimage.distance_transform_cdt(~edge_map,metric='taxicab')

    return dt

def distance_transform(mat,edge_sigma=1):
    """
    Computes the taxicab distance transform on binary array.
    """

    dt = ndimage.distance_transform_cdt(mat,metric='taxicab')

    return dt

def my_thr(img,rel_thr,sigma):

    img_thr = np.zeros((img.shape[0],img.shape[1]))
    if sigma is not None:
        img = filters.gaussian(img,sigma,multichannel=True)
    img_thr = color.rgb2gray(img)
    img_thr = (img_thr-np.mean(img_thr))/np.std(img_thr)
    high_thr = np.mean(img_thr)+rel_thr*np.std(img_thr)
    low_thr = np.mean(img_thr)-rel_thr*np.std(img_thr)
    img_thr[np.where(img_thr > high_thr)] = 0
    img_thr[np.where(img_thr < low_thr)] = 0

    return img_thr

def get_features_hough(imgs,rel_thr,max_n_lines, grid_step=1,sig_canny=1,radius=1,threshold=10, line_length=45,line_gap=3,labels=None):
    """
    This is a wrapper function on make_hough that processes a list of images.
    """

    w = imgs[0].shape[1]
    h = imgs[0].shape[0]

    hough_maps = [ make_hough(imgs[i],rel_thr, max_n_lines,sig_canny,radius,threshold,line_length,line_gap) for i in range(len(imgs))]
    if(labels is not None):
        X_hough = [img_crop_sp(hough_maps[i], labels[i]) for i in range(len(hough_maps))]
        X_hough = [np.any(X_hough[0][i]) for i in range(len(X_hough[0]))]
        return X_hough
    else:

        patches_per_image = [img_crop(hough_maps[i], grid_step, grid_step) for i in range(len(hough_maps))]
        patches = np.asarray([np.mean(patches_per_image[i][j].astype(int)) for i in range(len(patches_per_image)) for j in range(len(patches_per_image[i]))])
        return patches.reshape(-1,1)

def make_hough(img,rel_thr,max_n_lines,sig_canny=1,radius=1,threshold=10, line_length=45,line_gap=3):
    """
    Extracts Hough lines from image. Lines are sorted according to the RGB variance of pixels. max_n_lines are returned.
    """

    #Hough-lines extractor
    elem = morphology.disk(1)
    hough_lines = list()
    im = np.abs(my_thr(img,rel_thr,sig_canny))
    hough_lines.append(np.zeros((im.shape[0],im.shape[1])))

    lines = probabilistic_hough_line(im, threshold=threshold, line_length=line_length,line_gap=line_gap)
    line_idx = list()
    line_std_rgb = list()
    for line in lines:
        p0, p1 = line
        line_idx.append( draw.line(p0[1], p0[0], p1[1], p1[0]))
        line_std_rgb.append(np.std(color.rgb2gray(img)[line_idx[-1][0],line_idx[-1][1]]))
        #hough_lines[-1][line_idx] = 1
    line_std_rgb = np.asarray(line_std_rgb)
    line_std_idx = np.argsort(line_std_rgb)[0:max_n_lines]
    for i in range(np.min((max_n_lines,len(lines)))):
        this_line_idx = line_idx[line_std_idx[i]]
        hough_lines[-1][this_line_idx] = 1

    hough_lines[-1] = morphology.binary_dilation(hough_lines[-1],elem)

    return np.asarray(hough_lines).reshape(im.shape[0],im.shape[1])

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def kmeans_img(img,n_clusters):
    """
    Performs kmeans clustering (vector quantization) on input image using K=n_clusters.
    """

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(img.shape)
    assert d == 3
    image_array = np.reshape(img, (w * h, d))

    #print("Fitting model on a small sub-sample of the data")
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_array_sample)

    # Get labels for all points
    #print("Predicting color indices on the full image (k-means)")
    labels = kmeans.predict(image_array)
    #print("done in %0.3fs." % (time() - t0))

    codebook_random = shuffle(image_array, random_state=0)[:n_clusters + 1]
    #print("Predicting color indices on the full image (random)")
    labels_random = pairwise_distances_argmin(codebook_random,image_array, axis=0)

    return recreate_image(kmeans.cluster_centers_, labels, w, h)

def get_features_sift(imgs,canny_sigma,sift_sigmas,grid_step=16,return_kps=False,labels=None):
    """
    This is a wrapper function on get_sift_densely that processes a list of images.
    """

    w = imgs[0].shape[1]
    h = imgs[0].shape[0]

    if(labels is not None):
        X_sift = [ get_sift_densely(imgs[i],step=1,sigmas=sift_sigmas,mode='normal',return_kps=False) for i in range(len(imgs))]
        patches_per_image = [img_crop_sp(X_sift[i].reshape(h,w,X_sift[i].shape[1]), labels[i]) for i in range(len(X_sift))]
        return patches_per_image
    else:
        X_sift = np.asarray([ get_sift_densely(imgs[i],step=grid_step,sigmas=sift_sigmas,mode='normal',return_kps=False) for i in range(len(imgs))])
        X_sift = X_sift.reshape(len(imgs)*X_sift.shape[1],-1)

        return X_sift

def get_sift_densely(img,step=1,sigmas=None,mode='neighborhood',subsampl_step = 2,macro_length=2,return_kps=False):
    """
    Extract SIFT descriptors on a dense grid. The neighborhood option allows to concatenate several SIFT descriptor sampled around a keypoint. That option was tested but not retained (slow and non-effective).

    """

    def do_it(img,step):
        sift = cv2.xfeatures2d.SIFT_create(sigma=1)
        kpDense = [cv2.KeyPoint(x, y, step) for y in range(0, img.shape[0], step)  for x in range(0, img.shape[1], step)]
        img = (color.rgb2gray(img)*255).astype(np.uint8)
        kps,des = sift.compute(color.rgb2gray(img),kpDense)
        des = des/np.linalg.norm(des,axis=1).reshape(-1,1)
        des[np.where(des >= 0.2)] = 0.2
        des = des/np.linalg.norm(des,axis=1).reshape(-1,1)

        return des

    if((sigmas is None) and mode is not 'neighborhood'): return do_it(img,step)
    if((sigmas is None) and mode is 'neighborhood'):
        des = list()
        this_des = do_it(img,1).reshape(img.shape[0],img.shape[1],-1)
        centers = [(x,y) for y in np.arange(0, img.shape[0], step)  for x in np.arange(0, img.shape[1], step)]
        for i in range(len(centers)):
            idx_i = np.arange(centers[i][0]-macro_length*subsampl_step/2, centers[i][0] + macro_length*subsampl_step/2 + 1 , subsampl_step).astype(int)
            idx_j = np.arange(centers[i][1]-macro_length*subsampl_step/2, centers[i][1] + macro_length*subsampl_step/2 + 1 , subsampl_step).astype(int)
            des.append(this_des.take(idx_i,mode='wrap',axis=0).take(idx_j,mode='wrap',axis=1).ravel())

        res = np.asarray(des)
        if(return_kps):
            return(res,centers)
        else:
            return res

    else:
        des = list()
        for i in range(len(sigmas)):
            des.append(do_it(filters.gaussian(img,sigmas[i],multichannel=True),step))

        res = np.asarray(des)
        res = np.transpose(res,(1,2,0))

        return res.reshape(res.shape[0],res.shape[1]*res.shape[2])

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def rgb_remove_green(img):
    """
    Removes the green channel.
    """
    img[:,:,1] = np.zeros((img.shape[0],img.shape[1]))
    return img

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop_sp(im, labels):
    """
    Returns list of pixel values (array) contained in labels
    """

    h = im.shape[0]
    w = im.shape[1]
    patches = list()
    is_2d = len(im.shape) < 3
    for i in np.unique(labels):
        this_mask = np.where(labels == i)
        if is_2d:
            patches.append(im[this_mask[0],this_mask[1]])
        else:
            patches.append(im[this_mask[0],this_mask[1],:])

    return patches

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_more_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image
def extract_img_features(filename,patch_size):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    return X

def value_to_class(v,thr):
    df = np.sum(v)
    if df > thr:
        return 1
    else:
        return 0

def sp_label_to_img(labels, Z):
    im = np.zeros((labels.shape[0], labels.shape[1]))
    for i in range(Z.shape[0]):
        if(Z[i] == 1):
            im += labels == i
    return im

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 4)

def prepare_data(X):
    X_directions = []
    X_edge_features = []
    for x in X:
        # get edges in grid
        right, down = make_grid_edges(x, return_lists=True)
        edges = np.vstack([right, down])
        # use 3x3 patch around each point
        features = x.reshape(x.shape[0]*x.shape[1],-1)
        # simple edge feature that encodes just if an edge is horizontal or
        # vertical
        edge_features_directions = edge_list_to_features([right, down])
        X_directions.append((features, edges, edge_features_directions))
    return X_directions

def make_features_sp(img,pca,canny_sigma,slic_comp,slic_segments,hough_rel_thr,hough_max_lines,hough_canny, hough_radius, hough_threshold, hough_line_length, hough_line_gap,codebook):

    """
    Extracts features on superpixel-segmented image.
    """
    sift_bow = list()
    labels = segmentation.slic(img, compactness=slic_comp, n_segments=slic_segments)
    X_hough = np.asarray(get_features_hough([img],hough_rel_thr,hough_max_lines, 1,hough_canny,hough_radius,hough_threshold , hough_line_length,hough_line_gap,labels = [labels]))
    sift_bow.append([])
    X_sift = np.asarray(get_features_sift([img],canny_sigma,None,labels=[labels]))
    X_sift = [pca.transform(X_sift[0][i].reshape(-1,128)) for i in range(len(X_sift[0]))]
    for j in range(len(X_sift)):
        code_sift, dist = cluster.vq.vq(X_sift[j], codebook)
        sift_bow[-1].append(np.histogram(code_sift,bins=codebook.shape[0],density=True)[0])
    X_sift = np.asarray(sift_bow[-1])
    X_rgb = np.asarray(get_features_rgb([img],labels=[labels]))
    X_dt = np.asarray(get_features_dt([img],canny_sigma,labels = [labels]))
    X = np.concatenate((X_hough.reshape(-1,1),X_dt.reshape(-1,1),X_rgb,X_sift),axis=1)

    return X, labels

"""----------------------tat helpers extra----------------------------------"""
def img_crop2(im, w, h, overlap = 0):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in np.arange(0,imgheight,h*(1-overlap)):
        for j in np.arange(0,imgwidth,w*(1-overlap)):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def compute_feats(image, kernels):
	feats = np.zeros((len(kernels)*2, 1), dtype=np.float32)
	for k, kernel in enumerate(kernels):
		filtered = ndimage.convolve(image, kernel, mode='constant') #,mode='wrap'
		feats[k*2] = filtered.mean()
		feats[k*2+1] = filtered.var()
	return feats #maybe also filtered ?

def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i

def get_gabor_kernels(theta_range, sigma_range, freq_range):
	kernels = []
	for theta in theta_range:
		theta = theta / 4. * np.pi
		for sigma in sigma_range:
			for frequency in freq_range:
				kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
				kernels.append(kernel)
	return kernels
