import cv2
from filters import utils, kernelize as k
import numpy as np
import astropy.wcs as a

filepath = '../data/3s.fits'
img = utils.get_data(filepath)

niter = 1
blockSize = 13
const = -10

while True:
	key = cv2.waitKey(0)
	if key == 27:  # esc
		break
	elif key == 82:  # up
		blockSize += 2
	elif key == 84 and blockSize >= 5:  # down
		blockSize -= 2
	elif key == 83:  # right
		const += 1
	elif key == 81:  # left
		const -= 1
	elif key == 103:  # g
		niter += 1
	elif key == 102 and niter >= 1:  # f
		niter -= 1
	else:
		print(key)

	smoothed = k.gaussian_median(img, 3, 7, niter)
	segmented = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, const)

	#mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

	# Set up the SimpleBlobdetector with default parameters.
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = 0
	params.maxThreshold = 256

	# Filter by Area.
	params.filterByArea = True
	params.minArea = 30

	# Filter by Circularity
	params.filterByCircularity = True
	params.minCircularity = 0.1

	# Filter by Convexity
	params.filterByConvexity = False
	params.minConvexity = 0.5

	# Filter by Inertia
	params.filterByInertia = False
	params.minInertiaRatio = 0.5

	detector = cv2.SimpleBlobDetector_create(params)

	# Detect blobs.
	reversemask = 255 - segmented
	keypoints = detector.detect(reversemask)

	im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	wcs = a.WCS(filepath)

	for keyPoint in keypoints:
		print('----------------------------------')
		print("Pixel Cordinates: {0}".format(keyPoint.pt))
		print("Radius: {0}".format(keyPoint.size))
		ra, dec = wcs.wcs_pix2world(keyPoint.pt[0], keyPoint.pt[1], 0)
		print("RA,DEC: ({0},{1})".format(ra, dec))
		radius = int(keyPoint.size/2)
		mask = np.zeros(img.shape)
		ykp = int(keyPoint.pt[0]) #why?
		xkp = int(keyPoint.pt[1])
		for i in range(-radius, radius):
			for j in range(-radius, radius):
				mask[xkp + i, ykp + j] = 1
		#salvare maschere in array
	#moltiplicare maschera x smoothed
	#cercare il massimo + neighbour al n%


	print('======================================blockSize, const =' + str([blockSize, const]))

	utils.show2(Original=im_with_keypoints, Smoothed=smoothed, Dilated=mask)
