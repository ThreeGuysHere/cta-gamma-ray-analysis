from astropy.io import fits
import cv2
import numpy as np


def getGaussianKernel(ksize,sigma=-1):
     return cv2.getGaussianKernel(ksize, sigma) * np.matrix.transpose(cv2.getGaussianKernel(ksize, sigma))


def gaussian_median(src, gksize, mksize, nsteps):

    output = src.copy()
    gaussian_kernel = getGaussianKernel(gksize)

    for i in range(nsteps):
        output = cv2.filter2D(output, -1, gaussian_kernel)
        output = cv2.medianBlur(output, mksize)

    cv2.normalize(output, output, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return output

