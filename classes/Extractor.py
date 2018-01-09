from filters import kernelize as k, utils, centroid_extraction as ce
from classes import TimeChecker as timer, BlobResult as blob
import cv2
import numpy as np
from io import StringIO


class Extractor:

	def __init__(self, fits_path=None, xml_input_path=None):
		"""
		Constructor
		"""
		self.fits_path = fits_path
		self.xml_input_path = xml_input_path

		# instanzia il lettore
		self.median_iter = 1
		self.median_ksize = 7
		self.gaussian_iter = 1
		self.gaussian_ksize = 3

		self.local_mode = 0
		self.threshold_mode = 0

		self.local_stretch_ksize = 21
		self.local_stretch_step_size = 5
		self.local_stretch_min_bins = 5

		self.local_eq_ksize = 15
		self.local_eq_clip_limit = 2.0

		self.AD_block_size = 13
		self.AD_const = -7

		self.print_intermediate = True

		print('Extractor initialised')
		return

	def perform_extraction(self):
		"""
		Identifies the gamma-ray source coordinates (RA,Dec) from the input map and creates a xml file for ctlike
		:param local: TODO
		:param adaptive: TODO
		:return: (ra,dec) coordinates of the source and the path of the output xml
		"""

		time = timer.TimeChecker()
		# Open fits map
		img = utils.get_data(self.fits_path)
		print("loaded map: {0}".format(self.fits_path))

		time.toggle_time("read")

		# Filter map

		smoothed = k.median_gaussian(img, self.median_iter, self.median_ksize, self.gaussian_iter, self.gaussian_ksize)

		if self.local_mode:
			localled = self.local_stretching(smoothed, self.local_stretch_ksize, self.local_stretch_step_size, self.local_stretch_min_bins)
		else:
			localled = self.local_equalization(smoothed, self.local_eq_ksize, self.local_eq_clip_limit)

		time.toggle_time("smoothing")

		# Binary segmentation and binary morphology
		# TODO: tuning parametri per condizioni variabili
		if self.threshold_mode:
			segmented = cv2.adaptiveThreshold(localled, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.AD_block_size, self.AD_const)
		else:
			segmented = cv2.threshold(localled, 127, 255, cv2.THRESH_BINARY)[1]
		# TODO: sorgenti grandi "da riempire": dilation
		# TODO: sorgenti piccole "deboli": dilation (attenzione a non eliminarle con erosion)
		# TODO: sorgenti sovrapposte: erosion (ma attenzione a quanta, si rischia di eliminare sorgenti deboli)
		# TODO: rumore spurio: erosion ma stesso discorso di sopra (o median filter a posteriori)
		segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
		# segmented = cv2.erode(segmented, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)), iterations=2)
		# segmented = cv2.dilate(segmented, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)

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

		# Create output file
		index = 0
		buffer = StringIO()

		for keyPoint in keypoints:
			print('----------------------------------')
			current_blob = blob.BlobResult(self.fits_path, index)
			index = index + 1

			current_blob.set_bary(keyPoint.pt)
			current_blob.set_diameter(keyPoint.size)
			current_blob.set_mask(img.shape)
			current_blob.print_values()

			buffer.write(current_blob.make_xml_blob())
		print('=================================')

		time.toggle_time("blob extraction")
		time.total()

		utils.show2(Blobbed=im_with_keypoints, Original=img)

		return utils.create_xml(buffer.getvalue())

	def local_stretching(self, smoothed, ksize=21, step_size=5, min_bins=1):
		# Filter map
		localled = smoothed.copy()

		for (x, y, window) in utils.sliding_window(smoothed, stepSize=step_size, windowSize=(ksize, ksize)):
			local_hist = cv2.calcHist([window], [0], None, [256], [0, 255])
			bins = np.count_nonzero(local_hist)
			if bins > min_bins:
				window1 = window.copy()
				cv2.normalize(window1, window1, 0, 255, cv2.NORM_MINMAX)
				localled[y:y + ksize, x:x + ksize] = window1

		if self.print_intermediate:
			utils.show(Smoothed=smoothed, Local=localled)
		return localled

	def local_equalization(self, smoothed, ksize=15, clip_limit=2.0):
		# Filter map
		clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(ksize, ksize))
		localled = clahe.apply(smoothed)

		if self.print_intermediate:
			utils.show(Smoothed=smoothed, Local=localled)
		return localled