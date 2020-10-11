import unet as model
import cv2 as cv

"""
run unet to make predict on ONE image
"""

IMG_PATH = '/home/xin/comp9517/proj/data/images/train-volume01.jpg'
RESULT_PATH = '/home/xin/comp9517/proj/results'
MODEL_PATH = '/home/xin/comp9517/proj/checkpoint/'
MODEL_NAME = 'unet.ckpt'

img_list = [cv.imread(IMG_PATH, 0)]
model.run(img_list, RESULT_PATH, MODEL_PATH, MODEL_NAME)

