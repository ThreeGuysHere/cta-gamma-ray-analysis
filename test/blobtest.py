from classes import Extractor

fits_name = '../img/3s_displaced.fits'
ext = Extractor.Extractor(fits_name)
xml = ext.perform_extraction()
print("output_xml_path = {0}".format(xml))
