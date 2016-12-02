# pcml_p2
## Requirements (LL)
- For the computation of SIFT descriptors, install opencv as well as opencv_contrib as explained here:
http://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/
##See report.pdf for litterature review
## Changelogs and other various things
##LL 27/11: Added DAISY feature descriptor (similar to SIFT) on RGB channels
##LL 2/12: Class rebalancing (oversampling of positive class). Also, features extraction now follows the bag-of-visual-words procedure:
- dense-SIFT + HSV colors computation
- PCA
- K-means (codebook) 
- Computation of histograms (bag-of-words) on patches
