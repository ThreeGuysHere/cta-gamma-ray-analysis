from astropy.io import fits
import numpy as np
import cv2

#  screen info
screen_width = 1650
screen_height = 1050
dx = int(screen_width / 3)
dy = int(screen_height / 2)
label_bar_height = 65


def img_prepare(src):
	cv2.normalize(src, src, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
	return src.astype(np.uint8)


def normalize(src):
	src_max = np.max(src)
	src_min = np.min(src)
	src = np.multiply(src-src_min, 255/(src_max - src_min))
	return src


def get_data(src):
	# Overflow aware version
	# data = fits.getdata(src)
	# data = normalize(data)
	# return data.astype(np.uint8)

	# TRUNC version
	data = fits.getdata(src)
	img = data.astype(np.uint8)
	return img_prepare(img)


def show(**kwargs):
	x = 0
	y = 0
	for key in kwargs:
		label = key
		img = kwargs[key]
		cv2.namedWindow(label, cv2.WINDOW_NORMAL)
		cv2.imshow(label, img)
		cv2.resizeWindow(label, dx, dy - label_bar_height)
		cv2.moveWindow(label, x, y)

		screen_end = int(x / (screen_width - dx)) > 0

		x = x + dx if not screen_end else 0
		y += dy if screen_end else 0

	cv2.waitKey(0)
	cv2.destroyAllWindows()


def show2(**kwargs):
	x = 0
	y = 0
	for key in kwargs:
		label = key
		img = kwargs[key]
		cv2.namedWindow(label, cv2.WINDOW_NORMAL)
		cv2.imshow(label, img)
		cv2.resizeWindow(label, dx, dy - label_bar_height)
		cv2.moveWindow(label, x, y)

		screen_end = int(x / (screen_width - dx)) > 0

		x = x + dx if not screen_end else 0
		y += dy if screen_end else 0


def create_xml(model, output, pnt_type="equatorial", ra=83.64, dec=22.02, obsid=1, start=0.0, duration=100.0,
				emin=0.1, emax=100, caldb="prod2", fov=10, irf="South_0.5h"):
	with open(model, 'r') as model_xml:
		with open(output, 'w') as output_file:

			# read model
			parametrized = model_xml.read()

			# replace params
			parametrized = parametrized.replace("PNT", str(pnt_type))
			parametrized = parametrized.replace("RAX", str(ra))
			parametrized = parametrized.replace("DEC", str(dec))
			parametrized = parametrized.replace("OBS", str(obsid))
			parametrized = parametrized.replace("STR", str(start))
			parametrized = parametrized.replace("DUR", str(duration))
			parametrized = parametrized.replace("EMI", str(emin))
			parametrized = parametrized.replace("EMA", str(emax))
			parametrized = parametrized.replace("CAL", str(caldb))
			parametrized = parametrized.replace("FOV", str(fov))
			parametrized = parametrized.replace("IRF", str(irf))

			# write replaced
			output_file.write(parametrized)

	return parametrized


def plot3d(src):
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import pylab as pl
	from PIL import Image
	import numpy as np

	img = Image.fromarray(src)

	z = np.asarray(img)
	mydata = z[::1, ::1]
	fig = pl.figure(facecolor='w')
	ax1 = fig.add_subplot(1, 2, 1)
	im = ax1.imshow(mydata, interpolation='nearest', cmap=pl.cm.jet)
	ax1.set_title('2D')

	ax2 = fig.add_subplot(1, 2, 2, projection='3d')
	x, y = np.mgrid[:mydata.shape[0], :mydata.shape[1]]
	ax2.plot_surface(x, y, mydata, cmap=pl.cm.jet, rstride=1, cstride=1, linewidth=0., antialiased=False)
	ax2.set_title('3D')
	ax2.set_zlim3d(0, 255)
	pl.show()


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0] - int(windowSize[0]) + 1, stepSize):
		for x in range(0, image.shape[1] - int(windowSize[1]) + 1, stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
