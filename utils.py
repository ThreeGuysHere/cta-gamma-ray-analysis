from astropy.io import fits
import numpy as np
import cv2


def get_data(src):
    mapdata = fits.getdata(src)

    img = mapdata.astype(np.uint8)
    cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return img


def img_prepare(src):
    cv2.normalize(src, src, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return src.astype(np.uint8)


def show(**kwargs):
    for key in kwargs:
        label = key
        cv2.namedWindow(label, cv2.WINDOW_NORMAL)
        cv2.imshow(label, kwargs[key])
        cv2.resizeWindow(label, 500, 500)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
