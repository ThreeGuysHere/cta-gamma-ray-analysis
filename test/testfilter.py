import cv2
import filter
import utils
import centroid_extraction as ce
import kernelize as k
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

img = utils.get_data('../data/cube.fits')

output = filter.gaussian_median(img, 3, 15, 3)

mask = cv2.threshold(output, 1, 1, cv2.THRESH_BINARY)[1]
print(ce.find_weighted_centroid(output, mask))
print(ce.find_barycenter(mask))

img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
utils.show(Original=img, Result=output)

