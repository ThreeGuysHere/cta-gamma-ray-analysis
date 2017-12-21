from astropy.io import fits
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

map = fits.getdata('map.fits')

img = map.astype(np.uint8)
cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
output = img.copy()

cv2.imshow('original', output)

gaussian_kernel = np.matrix('1 2 1; 2 4 2; 1 2 1') / 9.0

for i in range(2):
    output = cv2.filter2D(output, -1, gaussian_kernel)
    output = cv2.medianBlur(output, 15)

output = cv2.applyColorMap(output.astype(np.uint8), 11)

cv2.imshow("original", img)
cv2.imshow("output", output)

# plt.imshow(output, interpolation='nearest')
# plt.show()
cv2.waitKey(0)

