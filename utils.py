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


def show(img1, img2):
    src = "Original"
    out = "Output"
    cv2.namedWindow(src, cv2.WINDOW_NORMAL)
    cv2.namedWindow(out, cv2.WINDOW_NORMAL)
    cv2.imshow(src, img1)
    cv2.resizeWindow(src, 500, 500)
    cv2.moveWindow(src, 0, 0)
    cv2.imshow(out, img2)
    cv2.moveWindow(out, 500, 0)
    cv2.resizeWindow(out, 500, 500)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

