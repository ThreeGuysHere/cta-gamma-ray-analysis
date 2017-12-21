import numpy as np
import math

def ker(map):
    '''
    computes the kernel mask
    :param map: ndarray, the raw fits map
    :return: boh
    '''

    print(map.shape)
    maxIntensity = np.max(map)

    coords = np.unravel_index(map.argmax(), map.shape)

    cazzi = np.matrix([coords[0], coords[1], map.shape[0]-coords[0], map.shape[1]-coords[1]])
    ksize = np.matrix.min(cazzi)

    print(ksize)
    print("max coord = {0}".format(coords))
    print("255-247 = {0}".format(map[255,247]))
    print("247-255 = {0}".format(map[247,255]))
    return maxIntensity