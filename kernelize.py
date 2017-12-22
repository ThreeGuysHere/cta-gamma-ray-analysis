import numpy as np
import filter as f
import cv2


def gray_gaussian_mask_at(img, coords, normalize=False):
    """
    Computes the kernel mask centered in the highest intensity points
    :param img: ndarray, the raw fits map
    :param coords: coordinates of the gaussian curve center
    :param normalize: if true, it normalizes the matrix (0,255)
    :return: the gaussian mask gray matrix
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
    if normalize:
        cv2.normalize(mask, mask, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return mask.astype(np.uint8)


def rgb_gaussian_mask_at(img, coords, normalize=False):
    """
    Computes the kernel mask centered in the highest intensity points
    :param img: ndarray, the raw fits map
    :param coords: coordinates of the gaussian curve center
    :param normalize: if true, it normalizes the matrix (0,255)
    :return: the gaussian mask rgb matrix
    """
    return cv2.cvtColor(gray_gaussian_mask_at(img, coords, normalize), cv2.COLOR_GRAY2RGB)


def gaussian_mask(img):
    coords = np.argwhere(img == np.amax(img))
    coords = [(i[0], i[1]) for i in coords[:,:-1]]
    # print(coords)
    maps = []
    if isinstance(coords, list):
        for c in coords:
            maps.append(rgb_gaussian_mask_at(img, c, True))
            cv2.imshow(str(c), maps[coords.index(c)])
    else:
        maps.append(rgb_gaussian_mask_at(img, coords, True))

    return maps
