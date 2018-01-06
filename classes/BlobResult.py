from astropy.wcs import WCS
import numpy as np


class BlobResult:

	def __init__(self, fits_path):
		"""
		Constructor
		"""
		self.path_map = fits_path
		self.bary = None
		self.diam = None
		self.radius = None
		self.radec = None
		self.mask = None
		return

	def set_bary(self, barycenter):
		self.bary = barycenter
		wcs = WCS(self.path_map)
		ra, dec = wcs.wcs_pix2world(self.bary[0], self.bary[1], 0)
		self.radec = [ra, dec]
		return

	def set_diameter(self, size):
		self.diam = size
		self.radius = int(size/2)
		return

	def set_mask(self, shape):
		msk = np.zeros(shape)
		ykp = int(self.bary[0])  # why?
		xkp = int(self.bary[1])
		for i in range(-self.radius, self.radius):
			for j in range(-self.radius, self.radius):
				msk[xkp + i, ykp + j] = 1
		self.mask = msk
		return

	def print_values(self):
		print("Barycenter: {0}".format(self.bary))
		print("Size: {0}".format(self.diam))
		print("RA,Dec: ({0},{1})".format(self.radec[0], self.radec[1]))
		return
