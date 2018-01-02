import cv2
from filters import utils, kernelize as k
from matplotlib import pyplot as plt

img = utils.get_data('../data/3s.fits')

output = k.gaussian_median(img, 3, 7, 1)

#img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
#output = cv2.applyColorMap(output, cv2.COLORMAP_JET)

# output = cv2.threshold(output,127,255,cv2.THRESH_BINARY)[1]

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

	out = cv2.adaptiveThreshold(output, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ksize, boh)
	print([ksize, boh])
	utils.show2(Original=img, Result=out)

cv2.destroyAllWindows()

