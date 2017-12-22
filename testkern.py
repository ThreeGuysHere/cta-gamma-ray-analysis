import cv2
import filter
import utils
import kernelize as k
import numpy as np

img = cv2.imread('castello3.JPG')
out = k.gaussian_mask(img)
out = cv2.bitwise_and(img, out)
utils.show(img, out)

