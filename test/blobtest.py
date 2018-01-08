from classes import Extractor

fits_names = ['../img/1s.fits', '../img/1s_bis.fits', '../img/1s_noise.fits',
			  '../img/2s_strongweak.fits', '../img/3s_displaced.fits', '../img/3s_joined.fits',
			  '../img/3s_strong_noise.fits', '../img/3s_strongweak.fits', '../img/4s_strongweak.fits']

for x in range(9):
	ext = Extractor.Extractor(fits_names[x])
	xml = ext.perform_extraction()
	print("output_xml_path = {0}".format(xml))
