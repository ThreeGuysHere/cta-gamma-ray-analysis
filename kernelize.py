import numpy as np
import filter as f
import cv2

def ker(map):
    '''
    computes the kernel mask
    :param map: ndarray, the raw fits map
    :return: boh
    '''

    coords = np.unravel_index(map.argmax(), map.shape)

    distances = np.matrix([coords[0], coords[1], map.shape[0]-coords[0]-1, map.shape[1]-coords[1]-1])
    print("distances = {0}".format(distances))
    max = np.matrix.max(distances)
    print("max = {0}".format(max))

    ksize = 2*max+1
    kernel = f.getGaussianKernel(ksize)
    print("kernel_shape = {0}".format(kernel.shape))

    kernel2 = kernel[max-coords[0]:max+(map.shape[0] - coords[0]), max-coords[1]:max+(map.shape[1] - coords[1])]

    print("kernel2_shape = {0}".format(kernel2.shape))
    cv2.normalize(kernel, kernel, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(kernel2, kernel2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return kernel, kernel2

