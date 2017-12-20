from astropy.io import fits
import cv2
import func
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#map = fits.getdata('map.fits')
#img = np.zeros(map.shape)
img = cv2.imread('prova.png', cv2.IMREAD_GRAYSCALE)

# for x in range(500):
#     for y in range(500):
#         if img[x, y] > 0:
#             img[x, y] = 127


cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
cv2.imshow('original', img)

gaussian_kernel = np.matrix('1 2 1; 2 4 2; 1 2 1') / 9.0

for i in range(2):
    img = cv2.filter2D(img, -1, gaussian_kernel)
    img = cv2.medianBlur(img,13)


img = cv2.applyColorMap(img.astype(np.uint8), 11)

# img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]

cv2.imshow('smoothed', img)
cv2.waitKey()