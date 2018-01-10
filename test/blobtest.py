from classes import Extractor
import glob
import cv2


def print_mode():
	print("\nSelect mode: \n"
		  "f:\t\tgaussian & median filter\n"
		  "t:\t\tadaptive threshold\n"
		  "p:\t\tprint info\n"
		  "<- -> to change map"
		  "esc:\tquit")
	return


fits_names = glob.glob("../img/*.fits")

keys = {
	'right_arrow': 83,
	'left_arrow': 81,
	'up_arrow': 82,
	'down_arrow': 84,
	'esc': 27,
	'enter': 13,
	't': 116,
	'f': 102,
	's': 105,
	'e': 101,
	'p': 112,
	'1': 49,
	'2': 50,
	'3': 51,
	'4': 52,
	'5': 53,
}

mode = {
	'init': -1,
	'none': 0,
	'threshold': 1,
	'filter': 2,
	'equalization': 3,
	'stretch': 4,
}

index = len(fits_names) - 1
selected_mode = -1
selected_param = -1
run = True

ext = Extractor.Extractor(fits_names[index])
ext.load_config("../data/cta-config.xml")

while True:

	if run:
		ext.perform_extraction()
		ext.prints = False

	if selected_mode == mode['init']:
		print_mode()
		selected_mode = mode['none']

	key = cv2.waitKey(0)

	if key == keys['esc']:
		break

	if key == keys['enter']:
		print_mode()

	# MODE
	elif key == keys['t']:
		selected_mode = mode['threshold']
		print("\nSelect parameter: (press t to get current value) \n"
				"1:\tadaptive kernel size\t{0}\n"
				"2:\tadaptive mean constant\t{1}\n".format(ext.adaptive_block_size,ext.adaptive_const))

	elif key == keys['f']:
		selected_mode = mode['filter']
		print("\nSelect parameter: (press f to get current value) \n"
				"1:\tnumber of median iterations\t\t{0}\n"
				"2:\tmedian kernel size\t\t\t\t{1}\n"
				"3:\tnumber of gaussian iterations\t{2}\n"
				"4:\tgaussian kernel size\t\t\t{3}\n".format(ext.median_iter,ext.median_ksize,ext.gaussian_iter,ext.gaussian_ksize))

	elif key == keys['e']:
		selected_mode = mode['equalization']
		print("\nSelect parameter: (press e to get current value) \n"
				"1:\tequalization kernel size\t{0}\n"
				"2:\tclip limit\t\t{1}\n".format(ext.local_eq_ksize, ext.local_eq_clip_limit))

	elif key == keys['s']:
		selected_mode = mode['stretch']
		selected_param = 0
		print("\nSelect parameter: (press s to get current value) \n"
				"1:\tstretch kernel size\t{0}\n"
				"2:\tstretch step size\t{1}\n"
				"3:\tstretch min bins\t{2}\n".format(ext.local_stretch_ksize,ext.local_stretch_step_size,ext.local_stretch_min_bins))

	elif key == keys['p']:
		print("\n\n===============INFO===============")
		ext.prints = True
		run = True

	# FITS maps
	elif key in [keys['up_arrow'], keys['down_arrow']]:

		if key == keys['up_arrow']:
			index += 1
		elif key == keys['down_arrow']:
			index -= 1

		if index < len(fits_names) - 1 or index > 0:
			ext = Extractor.Extractor(fits_names[index])
			ext.load_config("../data/cta-config.xml")
			run = True
		else:
			run = False

	# PARAM
	elif key in [keys['1'], keys['2'], keys['3'], keys['4'], keys['5']]:
		if selected_mode != mode['none']:
			selected_param = int([k for k, v in keys.items() if v == key][0])
		else:
			print("\nNo mode selected!")
	elif key in [keys['right_arrow'], keys['left_arrow']]:
		if selected_param:

			if key == keys['right_arrow']:
				sign = 1
			elif key == keys['left_arrow']:
				sign = -1

			if selected_mode == mode['threshold']:
				if selected_param == 1:
					ext.adaptive_block_size += 2*sign
				elif selected_param == 2:
					ext.adaptive_const += sign

			elif selected_mode == mode['filter']:
				if selected_param == 1:
					ext.median_iter += sign
				elif selected_param == 2:
					ext.median_ksize += 2*sign
				elif selected_param == 3:
					ext.gaussian_iter += sign
				elif selected_param == 4:
					ext.gaussian_ksize += 2*sign

			elif selected_mode == mode['equalization']:
				if selected_param == 1:
					ext.local_eq_ksize += 2*sign
				elif selected_param == 2:
					ext.local_eq_clip_limit += sign

			elif selected_mode == mode['stretch']:
				if selected_param == 1:
					ext.local_stretch_ksize += 2*sign
				elif selected_param == 2:
					ext.local_stretch_step_size += sign
				elif selected_param == 3:
					ext.local_stretch_min_bins += sign
			else:
				print("\nNo mode selected!")
		else:
			print("\nNo parameter selected!")
		run = True
	else:
		run = False
		print(key)


# # cercare il massimo + neighbour al n%
# masked_original = np.multiply(smoothed, el.mask)
# center_intensity = np.argwhere(masked_original >= int(np.amax(masked_original)*0.95))
# print(center_intensity)

cv2.destroyAllWindows()


