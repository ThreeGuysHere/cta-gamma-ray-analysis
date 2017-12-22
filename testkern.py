import cv2
import filter
import utils
import kernelize as k
import numpy as np

img = cv2.imread('data/small.png', cv2.IMREAD_GRAYSCALE)
out = k.gaussian_mask(img)

res = np.multiply(img, out)

utils.show(kernel=utils.img_prepare(out), modificato=utils.img_prepare(res))

