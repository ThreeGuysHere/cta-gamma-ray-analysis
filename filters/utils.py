from astropy.io import fits
import numpy as np
import cv2

#  screen info
screen_width = 1650
screen_height = 1050
dx = int(screen_width/3)
dy = int(screen_height/2)
label_bar_height = 65


def img_prepare(src):
    cv2.normalize(src, src, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return src.astype(np.uint8)


def get_data(src):
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
        cv2.resizeWindow(label, dx, dy-label_bar_height)
        cv2.moveWindow(label, x, y)

        screen_end = int(x / (screen_width-dx)) > 0

        x = x + dx if not screen_end else 0
        y += dy if screen_end else 0

    cv2.waitKey(0)
    cv2.destroyAllWindows()
