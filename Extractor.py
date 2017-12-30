from filters import kernelize as k, utils, centroid_extraction as ce
import astropy.wcs as a
import cv2
import numpy as np


class Extractor:

	def __init__(self):
		"""
		Constructor
		"""
		# # vedere cosa vale la pena di mettere come campi privati

		# Set some standard test data
		# self._datadir = os.environ['PATH']
		# self.caldb = 'prod2'
		# self.irf = 'South_0.5h'

		print('Extractor initialised')
		return


	def perform_extraction(self, fits_map_path):
		"""
		Identifies the gamma-ray source coordinates (ra,dec) from the input map and creates a xml file for ctlike
		:param fits_map_path: input map.fits path
		:return: (ra,dec) coordinates of the source and the path of the output xml
		"""
		# open fits map
		img = utils.get_data(fits_map_path)
		print("loaded map: {0}".format(fits_map_path))

		# extraction
		output = k.gaussian_median(img, 3, 7, 1)

		mask = cv2.threshold(output, 127, 255, cv2.THRESH_BINARY)[1]

		# print(ce.find_weighted_centroid(output, mask))
		bar = ce.find_barycenter(mask)
		print("pixel coords = {0}".format(bar))

		# conversion pixel 2 (ra,dec)
		wcs = a.WCS(fits_map_path)
		ra_c,dec_c = wcs.wcs_pix2world(bar[0], bar[1], 0)
		print("ra={0}".format(ra_c))
		print("dec={0}".format(dec_c))
		print("verifica")
		x_c,y_c=wcs.wcs_world2pix(ra_c, dec_c, 1)
		print("x={0}".format(x_c))
		print("y={0}".format(y_c))

		# print(w.all_world2pix(self.in_ra, self.in_dec, 1))
		# print(w.all_pix2world(100, 100, 0))


		# print result
		img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
		output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
		utils.show(Original=img, Result=output)

		# create output xml file
		return [ra_c, dec_c], '/stay/tuned.xml'
