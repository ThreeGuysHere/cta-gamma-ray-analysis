import cv2
import filters.TimeChecker as t
from filters import utils, kernelize as k
from matplotlib import pyplot as plt
import numpy as np
import astropy.wcs as a


time = t.TimeChecker()
fits_name = '../data/3s.fits'
img = utils.get_data(fits_name)
time.mark_time("read")

smoothed = k.gaussian_median(img, 3, 7, 1)
time.mark_time("gaussian_median")

output = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -20)
mask = cv2.dilate(output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
time.mark_time("binarization")

# noise removal
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations = 1)


# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 256;

# Filter by Area.
params.filterByArea = True
params.minArea = 30

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.5

detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
reversemask = 255 - mask
keypoints = detector.detect(reversemask)

time.mark_time("blob analysis")


print(keypoints)
im_with_keypoints = cv2.drawKeypoints(smoothed, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

wcs = a.WCS(fits_name)

for keyPoint in keypoints:
    print('----------------------------------')
    print("Pixel Cordinates: {0}".format(keyPoint.pt))
    print("Radius: {0}".format(keyPoint.size))
    ra, dec = wcs.wcs_pix2world(keyPoint.pt[0], keyPoint.pt[1], 0)
    print("RA,DEC: ({0},{1})".format(ra, dec))
print('----------------------------------')

time.mark_time("RA,DEC conversion")
time.total()

utils.show(Original=img, Smoothed=smoothed, Dilated=mask, Blobbed=im_with_keypoints)
