import cv2
import numpy as np
from filters import utils, kernelize as k
from matplotlib import pyplot as plt

img = utils.get_data('../data/3s.fits')

output = k.gaussian_median(img, 3, 7, 1)

output = np.multiply(output.astype(np.uint32)**0.5, 255**0.5).astype(np.uint8)

hist = cv2.calcHist([output], [0], None, [256], [0, 255])
plt.plot(hist)
plt.show()

#cv2.normalize(output, output, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
utils.show(Original=img, Result=output)

