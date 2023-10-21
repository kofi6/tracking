import numpy as np


def APCE(matrix=None):
    max = np.amax(matrix)
    min = np.amin(matrix)
    sum = 0
    for num in matrix:
        a = (num-min)**2
        sum += a
    apce = ((max-min)**2)/(sum/matrix.size)
    return apce


class Criterion(matrix=None):
    
