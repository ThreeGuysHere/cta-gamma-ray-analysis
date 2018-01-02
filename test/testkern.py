import cv2
from filters import utils, kernelize as k
import numpy as np


print("test script")
new = utils.create_xml('../data/model.xml', '../data/ciao.xml', ra=83.0, dec=22.0)
print(new)

img = cv2.imread('../data/castello3.JPG', cv2.IMREAD_GRAYSCALE)
out = k.gaussian_mask(img)

res = np.multiply(img, out)

utils.show(originale=img, kernel=utils.img_prepare(out), modificato=utils.img_prepare(res))

