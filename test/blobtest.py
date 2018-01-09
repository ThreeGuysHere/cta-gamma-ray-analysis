from classes import Extractor
import cv2

fits_names = ['../img/1s.fits', '../img/1s_bis.fits', '../img/1s_noise.fits',
			  '../img/2s_strongweak.fits', '../img/3s_displaced.fits', '../img/3s_joined.fits',
			  '../img/3s_strong_noise.fits', '../img/3s_strongweak.fits', '../img/4s_strongweak.fits']

index = 0
run = True

while True:
	if run:
		ext = Extractor.Extractor(fits_names[index])
		ext.load_config("../data/cta-config.xml")
		xml = ext.perform_extraction()

		print("output_xml_path = {0}".format(xml))

	key = cv2.waitKey(0)
	if key == 27:  # esc
		break
	elif key == 83 and index < len(fits_names)-1:  # right
		index += 1
		run = True
	elif key == 81 and index > 0:  # left
		index -= 1
		run = True
	else:
		run = False


# # cercare il massimo + neighbour al n%
# masked_original = np.multiply(smoothed, el.mask)
# center_intensity = np.argwhere(masked_original >= int(np.amax(masked_original)*0.95))
# print(center_intensity)

cv2.destroyAllWindows()
