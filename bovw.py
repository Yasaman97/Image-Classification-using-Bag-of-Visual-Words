import os
import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
import tensorflow as tf
%matplotlib inline
from matplotlib import pyplot as plt
from google.colab import drive
import torchvision.datasets as datasets
drive.mount('/content/drive') 
path = "/content/drive/My Drive/Datasets/VOC2007" 

os.chdir(path)
os.listdir(path)

annotation_path = path+"/VOCdevkit/VOC2007/ImageSets/Main/"
print(os.listdir(annotation_path))

image_path = path+"/VOCdevkit/VOC2007/JPEGImages/"
print(os.listdir(image_path))

#creating the sift function
sift = cv2.xfeatures2d.SIFT_create()


def load_images(file_name):
  """read the files, as especified by the desired category, 
    given to it under file_name, it returns a list of images in that file directory, 
    and a list of the numpy array of images."""

  line_cat = open(annotation_path+file_name).read().splitlines()
  word = [line_cat[i].split(" ") for i in range(len(line_cat))]
  images_cats = []
  for i in range(len(word)):
    if len(word[i])==2:
      tmp = word[i][1]
    else:
      tmp = word[i][2]
      if tmp == '1':
        images_cats.append(word[i][0])
  cats = []
  cats_nparray = []
  for i in range(len(images_cats)):
    img = cv2.imread(image_path + "/" + images_cats[i]+'.jpg')
    img = cv2.resize(img,(224, 224))
    cats_nparray.append(np.array(img).astype('float32'))
    cats.append(img)
  print(len(cats))
  return cats, cats_nparray


# Choose the category of images to be classified.
cats, cats_nparray = load_images('cat_trainval.txt')
dogs,dogs_nparray = load_images('dog_trainval.txt')
bicycles,bicycles_nparray = load_images('bicycle_trainval.txt')

# Splitting train and test sets
train_images = [{"key": 1, "value": cats[:round(0.8*len(cats))]},{"key": 2, "value": dogs[:round(0.8*len(dogs))]}, {"key": 3, "value": bicycles[:round(0.8*len(bicycles))]}]
test_images = [{"key": 1, "value": cats[round(0.8*len(cats)):]},{"key": 2, "value": dogs[round(0.8*len(dogs)):]}, {"key": 3, "value": bicycles[round(0.8*len(bicycles)):]}]

def sift_features(images):
  """ function extracts the keypoints and descriptors of the images using SIFT, 
  and returns first a list of all the descriptors, 
  and then a list of descriptors devided by their category. """  
  sift_vectors = {}
  descriptor_list = []
  for i in range(3):
    features = []
    for img in images[i]['value']:
      kp, des = sift.detectAndCompute(img,None)
      descriptor_list.extend(des)
      features.append(des)
    sift_vectors[i] = features
  return [descriptor_list, sift_vectors]

# Call the sift_features to extract the descriptors of training and test images
sifts = sift_features(train_images)
all_bovw_feature = sifts[1]
descriptor_list = sifts[0]

test_bovw_feature = sift_features(test_images)[1]  

def kmeans(k, descriptor_list):
  """ uses KMeans clustering algorithm, and after finding the center of descriptors, 
  returns these centers which are the visual words. """
  kmeans = KMeans(n_clusters = k, n_init=10)
  kmeans.fit(descriptor_list)
  visual_words = kmeans.cluster_centers_
  return visual_words

visual_words = kmeans(100, descriptor_list)

def find_index(image, center):
  """ The function is used in image_class in order to determine which 
    visual word each feature belongs to by calculating its distance from the centers."""
  count = 0
  ind = 0
  for i in range(len(center)):
    if(i == 0):
      count = distance.euclidean(image, center[i])
    else:
      dist = distance.euclidean(image, center[i])
      if(dist < count):
        ind = i
        count = dist
  return ind


def image_class(all_bovw, centers):
  """ The function takes the descriptors (features), and the center of the clusters (visual words) 
  and calculates the histograms of the frequency of the appearance of each feature in each image."""
  dict_feature = {}
  for key,value in all_bovw.items():
    category = []
    for img in value:
      histogram = np.zeros(len(centers))
      for each_feature in img:
        ind = find_index(each_feature, centers)
        histogram[ind] += 1
      category.append(histogram)
    dict_feature[key] = category
    plt.hist(histogram)
    plt.show()
  return dict_feature  


# BoVW for train set
bovw_train = image_class(all_bovw_feature, visual_words)
# BoVW for test set
bovw_test = image_class(test_bovw_feature, visual_words)


def testing(images, tests):
  """ The function takes the histograms of the training and test set, 
  uses the histograms of train set, and eucldean distance to determine which class 
  each test image belongs to, then calculates the number of true positive classifications."""  
  num_test = 0
  correct_predict = 0
  class_based = {}
  for test_key, test_val in tests.items():
    class_based[test_key] = [0, 0] # [correct, all]
    for tst in test_val:
      predict_start = 0
      minimum = 0
      key = "a" #predicted
      for train_key, train_val in images.items():
        for train in train_val:
          if(predict_start == 0):
            minimum = distance.euclidean(tst, train)
            key = train_key
            predict_start += 1
          else:
            dist = distance.euclidean(tst, train)

            if(dist < minimum):
              minimum = dist
              key = train_key
      if(test_key == key):
        correct_predict += 1
        class_based[test_key][0] += 1
      num_test += 1
      class_based[test_key][1] += 1

  return [num_test, correct_predict, class_based]

results_bowl = testing(bovw_train, bovw_test)

def accuracy(results):
  """The function uses the results of the classification function 
  to show the overal and class-based accuracy."""  
  avg_accuracy = (results[1] / results[0]) * 100
  print("Average accuracy: %" + str(avg_accuracy))
  print("\nClass based accuracies: \n")
  for key,value in results[2].items():
    acc = (value[0] / value[1]) * 100
    print(key ," : %" + str(acc))

accuracy(results_bowl)    