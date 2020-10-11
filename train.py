import unet as model
import cv2 as cv
import image_processing as imporc
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


"""
train the model
continue = False: train a new model, save the model to SAVE_PATH
continue = True: keep training the model on exist model at SAVE_PATH
"""


IMG_PATH = '/home/xin/comp9517/proj/data/augmentation/dev_samples/train'
LABEL_PATH = '/home/xin/comp9517/proj/data/augmentation/dev_labels/label'
WEIGHT_MAP_PATH = '/home/xin/comp9517/proj/data/augmentation/weight_maps/wm'
SAVE_PATH = '/home/xin/comp9517/proj/checkpoint/'
FILE_NAME = 'unet.ckpt'
CONTINUE = True

model.train(2000, impath=IMG_PATH, lbpath=LABEL_PATH, wmpath=WEIGHT_MAP_PATH, svpath=SAVE_PATH, file_name=FILE_NAME, cont=CONTINUE)



