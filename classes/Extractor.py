from filters import kernelize as k, utils, centroid_extraction as ce
from classes import TimeChecker as timer, BlobResult as blob
import cv2
import numpy as np
from io import StringIO
from bs4 import BeautifulSoup


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
		self.gaussian_sigma = -1

		self.local_mode = "stretching"
		self.threshold_mode = "adaptive"

		self.local_stretch_ksize = 15
		self.local_stretch_step_size = 5
		self.local_stretch_min_bins = 1

		self.local_eq_ksize = 15
		self.local_eq_clip_limit = 2.0

		self.adaptive_filtering = cv2.ADAPTIVE_THRESH_MEAN_C
		self.adaptive_block_size = 13
		self.adaptive_const = -7

		self.print_intermediate = False

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
		time.toggle_time("smoothing")

		# Contrast Enhancing
		localled = self.local_transformation(self.local_mode)(smoothed)
		time.toggle_time("contrast enhancing")

		# Binary segmentation and binary morphology
		segmented = self.segmentation(self.threshold_mode)(localled)

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

		utils.show(Blobbed=im_with_keypoints, Original=img)

		return utils.create_xml(buffer.getvalue())

	def local_stretching(self, img):
		return k.local_stretching(img, self.local_stretch_ksize, self.local_stretch_step_size, self.local_stretch_min_bins, self.print_intermediate)

	def local_equalization(self, img):
		return k.local_equalization(img, self.local_eq_ksize, self.local_eq_clip_limit, self.print_intermediate)

	def adaptiveThreshold(self, img):
		return cv2.adaptiveThreshold(img, 255, self.adaptive_filtering, cv2.THRESH_BINARY, self.adaptive_block_size, self.adaptive_const)

	def load_config(self, filepath):
		conf_file = open(filepath, "r")
		config = BeautifulSoup(conf_file, "html.parser")
		conf_file.close()
		root = config.gammarayanalysis
		if root.filtering.median:
			self.median_iter = utils.convert_node_value(root.filtering.median.iterations)
			self.median_ksize = utils.convert_node_value(root.filtering.median.kernelsize)
		if root.filtering.gaussian:
			self.gaussian_iter = utils.convert_node_value(root.filtering.gaussian.iterations)
			self.gaussian_ksize = utils.convert_node_value(root.filtering.gaussian.kernelsize)
			self.gaussian_sigma = utils.convert_node_value(root.filtering.gaussian.sigma)
		if root.localtransformation.type.string == "Stretching":
			self.local_mode = "stretching"
			self.local_stretch_ksize = utils.convert_node_value(root.localtransformation.kernelsize)
			self.local_stretch_step_size = utils.convert_node_value(root.localtransformation.stepsize)
			self.local_stretch_min_bins = utils.convert_node_value(root.localtransformation.minbins)
		elif root.localtransformation.type.string == "Equalization":
			self.local_mode = "equalization"
			self.local_eq_ksize = utils.convert_node_value(root.localtransformation.kernelsize)
			self.local_eq_clip_limit = utils.convert_node_value(root.localtransformation.stepsize)
		if root.segmentation.type.string == "Adaptive":
			self.threshold_mode = "adaptive"
			self.adaptive_filtering = self.adaptive_filter(utils.convert_node_value(root.segmentation.filter, "string")) #TODO lower

		return

	def local_transformation(self, x):
		return {
			"stretching": self.local_stretching,
			"equalization": self.local_equalization,
		}.get(x, utils.perror("local transformation"))

	def segmentation(self, x):
		return {
			"adaptive": self.adaptiveThreshold,
		}.get(x, utils.perror("segmentation"))

	def filter(self, x):
		return {
			"median": self.median_filtering,
			"gaussian": self.gaussian_filtering,
		}.get(x, utils.perror("filter"))

	def adaptive_filter(self, x):
		return {
			"Mean": cv2.ADAPTIVE_THRESH_MEAN_C,
			"Gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		}.get(x, utils.perror("adaptive filter"))

