import cv2
from filters import utils, kernelize as k
import numpy as np

img = utils.get_data('../data/3s.fits')

smoothed = k.gaussian_median(img, 3, 7, 1)

ksize = 21
boh = -20

while True:

	key = cv2.waitKey(0)
	if key == 27: #esc
		break
	elif key == 82:
		ksize += 2
	elif key == 84:
		ksize -= 2
	elif key == 83:
		boh += 1
	elif key == 81:
		boh -= 1
	else:
		print(key)

	out = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ksize, boh)
	print([ksize, boh]) # best 21, -5
	utils.show2(Original=img, Result=out)

cv2.destroyAllWindows()
