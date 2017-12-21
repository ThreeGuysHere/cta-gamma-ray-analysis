import cv2
import filter
import utils
import kernelize as k
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

img = utils.getdata('map.fits')

output = filter.gaussian_median(img, 3, 15, 3)

img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
output = cv2.applyColorMap(output, cv2.COLORMAP_JET)

# output = cv2.threshold(output,127,255,cv2.THRESH_BINARY)[1]

utils.show(img, output)

