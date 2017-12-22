import cv2
import filter
import utils
import kernelize as k
import numpy as np

img = cv2.imread('small2.png')
out = k.gaussian_mask(img)

out[0] = cv2.bitwise_and(img, out[0])
utils.show(img, out[0])

