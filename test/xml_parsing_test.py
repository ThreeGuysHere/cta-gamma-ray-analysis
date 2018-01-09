from classes import Extractor
import cv2

fits_names = ['../img/1s.fits', '../img/1s_bis.fits', '../img/1s_noise.fits',
			  '../img/2s_strongweak.fits', '../img/3s_displaced.fits', '../img/3s_joined.fits',
			  '../img/3s_strong_noise.fits', '../img/3s_strongweak.fits', '../img/4s_strongweak.fits']



ext = Extractor.Extractor(fits_names[1])

ext.load_config("../data/cta-config.xml")
ext.perform_extraction()

# xml = ext.perform_extraction()
cv2.destroyAllWindows()