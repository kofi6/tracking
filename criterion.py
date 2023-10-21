import numpy as np


def APCE(matrix=None):
    max = np.amax(matrix)
    min = np.amin(matrix)
    sum = 0
    for i in matrix:
        for j in i:
            a = (j-min)**2
            sum += a
    apce = ((max-min)**2)/(sum/matrix.size)
    return apce


def Criterion(matrix):
    apce = APCE(matrix)
    pv = np.amax(matrix)
    apceThreshold = 27
    if apce > apceThreshold:
        return True
    else:
        return False
