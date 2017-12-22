import cv2
import filter
import utils
import kernelize as k
import numpy as np

img = cv2.imread('data/castello3.JPG', cv2.IMREAD_GRAYSCALE)
out = k.gaussian_mask(img)

res = np.multiply(img, out)
cv2.normalize(res, res, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
culo = res.astype(np.uint8)

utils.show(img, culo)

