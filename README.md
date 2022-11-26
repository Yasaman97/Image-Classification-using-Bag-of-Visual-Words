## About

Using bag of "visual" words for image classification, which is an adaptation of BoW used in natural Language Processing. The sift function is used, in order to compute the keypoints and their descriptors of images. Then, using these keypoints and descriptors and a clustering algorithm, K-means, make clusters from descriptors; the centers of these clusters would be the "visual words". Next, the histogram of these visual words are computed, showing the frequency of each one of them in the set of images; these histograms are called BoVW (Bag of Visual Words). At last, using eucldean distance we determine which class each test image belongs to, then calculates the number of true positive classifications.