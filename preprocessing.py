from image_processing import data_augmentation
import cv2 as cv

"""
to generate training data
"""
img_list, label_list = [], []

IMG_NAME = '/home/xin/comp9517/proj/data/images/train-volume'
LABEL_NAME = '/home/xin/comp9517/proj/data/labels/train-labels'
SAVE_DIR = '/home/xin/comp9517/proj/data/augmentation'
NUM_IMAGE = 25 # num of img used to generate
QUANTITY = 3 # total number pairs to be generate(for square img) = (quantity + 1) * #img * 6
for i in range(0, NUM_IMAGE):
    img_list.append(cv.imread(IMG_NAME + str('%02d'%i) + '.jpg', 0))
    label_list.append(cv.imread(LABEL_NAME + str('%02d'%i) + '.jpg', 0))
_, _, _ = data_augmentation(img_list, label_list, quantity=QUANTITY, seed=10, SAVE_DIR=SAVE_DIR)
