import numpy as np
import cv2


def gray_gaussian_mask_at(img, coords):
    """
    Computes a kernel gaussian mask centered in the given coordinates of the gaussian curve center
    :param img: ndarray, the raw fits map
    :param coords: coordinates of the gaussian curve center
    :return: a gaussian mask gray matrix
    """
    distances = np.matrix([coords[0], coords[1], img.shape[0] - coords[0] - 1, img.shape[1] - coords[1] - 1])
    # print("distances = {0}".format(distances))
    max = np.matrix.max(distances)
    # print("max = {0}".format(max))

    ksize = 2*max+1

    kernel = get_gaussian_kernel(ksize)
    # print("kernel_shape = {0}".format(kernel.shape))

    mask = kernel[max-coords[0]:max+(img.shape[0] - coords[0]), max - coords[1]:max + (img.shape[1] - coords[1])]

    # print("mask_shape = {0}".format(mask.shape))
    return mask


def gaussian_mask(img):
    """
    Computes the kernel mask centered in the highest intensity points
    :param img: ndarray, the raw fits map
    :return: a gaussian mask gray matrix
    """
    coords = np.argwhere(img == np.amax(img))

    mask = np.zeros(img.shape)

    for c in coords:
        mask += gray_gaussian_mask_at(img, c)

    return mask


def get_gaussian_kernel(ksize, sigma=-1):
    """
    Returns a Gaussian Kernel ksize * ksize
    :param ksize: kernel size
    :param sigma: sigma value of the gaussian curve
    :return: gaussian kernel matrix
    """
    return cv2.getGaussianKernel(ksize, sigma) * np.matrix.transpose(cv2.getGaussianKernel(ksize, sigma))


def median_gaussian(src, median_iter=0, mksize=7, gaussian_iter=0, gksize=3):
    """
    Computes a gaussian and a medial 2D filter nsteps times.
    :param src: source image
    :param gksize: gaussian kernel side size
    :param mksize: medial kernel side size
    :param median_iter: number of median iterations
    :param gaussian_iter: number of gaussian iterations
    :return:
    """
    output = src.copy()
    gaussian_kernel = get_gaussian_kernel(gksize)

    for i in range(median_iter):
        output = cv2.medianBlur(src, mksize)
    for i in range(gaussian_iter):
        output = cv2.filter2D(src, -1, gaussian_kernel)

    # cv2.normalize(output, output, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return output

