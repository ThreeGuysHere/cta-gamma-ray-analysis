import cv2
import numpy as np
from filters import utils, kernelize as k
from matplotlib import pyplot as plt

filepath = '../img/3s_strong_noise.fits'
img = utils.get_data(filepath)

output = k.median_gaussian(img, 3, 7, 1)

output = np.multiply(output.astype(np.uint32)**0.5, 255**0.5).astype(np.uint8)

hist = cv2.calcHist([output], [0], None, [256], [0, 255])
plt.plot(hist)
plt.show()

#cv2.normalize(output, output, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
utils.show(Original=img, Result=output)

