import numpy as np
import math

def ker(map):
    '''
    computes the kernel mask
    :param map: ndarray, the raw fits map
    :return: boh
    '''

    coords = np.unravel_index(map.argmax(), map.shape)

    distances = np.matrix([coords[0], coords[1], map.shape[0]-coords[0], map.shape[1]-coords[1]])
    ksize = np.matrix.min(distances)

    print(ksize)


    return ksize