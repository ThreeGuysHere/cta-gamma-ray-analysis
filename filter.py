from astropy.io import fits
import cv2
import numpy as np


def get_gaussian_kernel(ksize,sigma=-1):
    """
    Returns a Gaussian Kernel ksize * ksize
    :param ksize: kernel size
    :param sigma: sigma value of the gaussian curve
    :return: gaussian kernel matrix
    """
    return cv2.getGaussianKernel(ksize, sigma) * np.matrix.transpose(cv2.getGaussianKernel(ksize, sigma))


def gaussian_median(src, gksize, mksize, nsteps):
    """
    Computes a gaussian and a medial 2D filter nsteps times.
    :param src: source image
    :param gksize: gaussian kernel side size
    :param mksize: medial kernel side size
    :param nsteps: number of repetitions
    :return:
    """
    output = src.copy()
    gaussian_kernel = get_gaussian_kernel(gksize)

    for i in range(nsteps):
        output = cv2.filter2D(output, -1, gaussian_kernel)
        output = cv2.medianBlur(output, mksize)

    cv2.normalize(output, output, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return output

