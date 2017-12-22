import numpy as np
import filter as f
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

    kernel = f.get_gaussian_kernel(ksize,200)
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
    coords = coords if len(img.shape) == 2 else coords[:, :-1]
    coords = [(i[0], i[1]) for i in coords]

    mask = np.zeros(img.shape[:2])
    if isinstance(coords, list):
        for c in coords:
            mask += gray_gaussian_mask_at(img, c)
    else:
        mask = gray_gaussian_mask_at(img, coords)

    cv2.normalize(mask, mask, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    mask = mask.astype(np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
