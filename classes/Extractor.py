from filters import kernelize as k, utils, centroid_extraction as ce
from classes import TimeChecker as timer, BlobResult as blob
import cv2
import numpy as np


class Extractor:

	def __init__(self, fits_path=None):
		"""
		Constructor
		"""
		self.fits_path = fits_path
		print('Extractor initialised')
		return

	def perform_extraction(self):
		"""
		Identifies the gamma-ray source coordinates (RA,Dec) from the input map and creates a xml file for ctlike
		:param fits_map_path: input map.fits path
		:return: (ra,dec) coordinates of the source and the path of the output xml
		"""
		time = timer.TimeChecker()
		# Open fits map
		img = utils.get_data(self.fits_path)
		print("loaded map: {0}".format(self.fits_path))

		time.toggle_time("read")

		# Filter map
		smoothed = k.gaussian_median(img, 3, 7, 1)
		localled = self.local_stretching()
		time.toggle_time("smoothing")

		# Binary segmentation and binary morphology
		block_size = 13
		const = -7
		# TODO: tuning parametri per condizioni variabili
		segmented = cv2.adaptiveThreshold(localled, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, const)

		# TODO: sorgenti grandi "da riempire": dilation
		# TODO: sorgenti piccole "deboli": dilation (attenzione a non eliminarle con erosion)
		# TODO: sorgenti sovrapposte: erosion (ma attenzione a quanta, si rischia di eliminare sorgenti deboli)
		# TODO: rumore spurio: erosion ma stesso discorso di sopra (o median filter a posteriori)
		segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
		#segmented = cv2.erode(segmented, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)), iterations=2)
		#segmented = cv2.dilate(segmented, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)

		time.toggle_time("segmentation")

		# Blob detection
		# # Detection parameters
		params = cv2.SimpleBlobDetector_Params()

		# # Thresholds
		params.minThreshold = 0
		params.maxThreshold = 256

		# # Filter blob by area
		params.filterByArea = True
		params.minArea = 15

		# # Filter blob by circularity
		params.filterByCircularity = True
		params.minCircularity = 0.2

		# # Remove unnecessary filters
		params.filterByConvexity = False
		params.filterByInertia = False

		detector = cv2.SimpleBlobDetector_create(params)

		# # Detect blobs
		reverse_segmented = 255 - segmented
		keypoints = detector.detect(reverse_segmented)

		# # Overlap keypoints and original image
		im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		blob_array = []
		for keyPoint in keypoints:
			current_blob = blob.BlobResult(self.fits_path)
			current_blob.set_bary(keyPoint.pt)
			current_blob.set_diameter(keyPoint.size)
			current_blob.set_mask(img.shape)

			blob_array.append(current_blob)

		time.toggle_time("blob extraction")

		for el in blob_array:
			print('----------------------------------')
			el.print_values()
			center, area, radius, masked = ce.find_weighted_centroid(smoothed, el.mask)
			el.bary = center
			el.diam = 2*radius
			print("center = {0}\narea = {1}\nradius = {2}\nRA,Dec = ({3},{4})".format(center, area, radius, el.radec[0], el.radec[1]))
			# cv2.imshow("Masks", masked)
			# cv2.waitKey()
		print('=================================')

		time.toggle_time("blob features")
		time.total()

		utils.show(Original=img, Smoothed=smoothed, Segmented=segmented, Blobbed=im_with_keypoints)


		# # cercare il massimo + neighbour al n%
		# masked_original = np.multiply(smoothed, el.mask)
		# center_intensity = np.argwhere(masked_original >= int(np.amax(masked_original)*0.95))
		# print(center_intensity)

		# create output xml file coi risultati
		return '/stay/tuned.xml'



	def local_stretching(self):
		time = timer.TimeChecker()
		# Open fits map
		img = utils.get_data(self.fits_path)
		print("loaded map: {0}".format(self.fits_path))

		time.toggle_time("read")

		# Filter map
		smoothed = k.gaussian_median(img, 3, 7, 1)
		localled = smoothed.copy()
		time.toggle_time("smoothing")

		ksize = 21
		for (x, y, window) in utils.sliding_window(smoothed, stepSize=5, windowSize=(ksize, ksize)):
			local_hist = cv2.calcHist([window], [0], None, [256], [0, 255])
			bins = np.count_nonzero(local_hist)
			if bins > 15:
				window1 = window.copy()
				cv2.normalize(window1, window1, 0, 255, cv2.NORM_MINMAX)
				localled[y:y + ksize, x:x + ksize] = window1
				# localled[y:y + ksize, x:x + ksize] = np.multiply(255/(rmax-rmin), (window-rmin))

		utils.show(Original=img, Smoothed=smoothed, Local=localled)
		return localled