import cv2
from filters import utils, kernelize as k
from matplotlib import pyplot as plt

img = utils.get_data('../data/3s.fits')

output = k.gaussian_median(img, 3, 7, 1)

plt.hist(img.ravel(), 256, [0, 256])
plt.show()

plt.hist(output.ravel(), 256, [0, 256])
plt.show()

#img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
#output = cv2.applyColorMap(output, cv2.COLORMAP_JET)

# output = cv2.threshold(output,127,255,cv2.THRESH_BINARY)[1]

utils.show(Original=img, Result=output)
