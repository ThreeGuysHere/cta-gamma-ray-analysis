import cv2
from filters import utils, kernelize as k
import numpy as np
import astropy.wcs as a

filepath = '../data/3s.fits'
img = utils.get_data(filepath)

output = img.copy()

ksize = 13
mean = -10

while True:
	key = cv2.waitKey(0)
	if key == 27: #esc
		break
	elif key == 82:  # up
		ksize += 2
	elif key == 84 and ksize>=5:  # down
		ksize -= 2
	elif key == 83:  # right
		mean += 1
	elif key == 81:  # left
		mean -= 1
	elif key == 103: # G
		output = k.gaussian_median(output, 3, 7, 1)
	else:
		print(key)

	output = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ksize, mean)

	mask = cv2.dilate(output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
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

	im_with_keypoints = cv2.drawKeypoints(output, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	wcs = a.WCS(filepath)

	for keyPoint in keypoints:
		print('----------------------------------')
		print("Pixel Cordinates: {0}".format(keyPoint.pt))
		print("Radius: {0}".format(keyPoint.size))
		ra, dec = wcs.wcs_pix2world(keyPoint.pt[0], keyPoint.pt[1], 0)
		print("RA,DEC: ({0},{1})".format(ra, dec))
	print('======================================ksize, mean =' + str([ksize, mean]))

	utils.show2(Original=img, Dilated=mask, Blobbed=im_with_keypoints)