import numpy as np
import math


# Time Feature
def iemg(vector):
    total_sum = 0.0
    for entry in vector:
        total_sum += abs(entry)
    return total_sum


def mav(vector):
    vector_array = np.array(vector)
    return sum(abs(vector_array)/len(vector_array))


def rms(vector):
    total_sum = 0.0
    for i in range(0, len(vector)):
        total_sum += vector[i]*vector[i]
    sigma = math.sqrt(total_sum/(len(vector)))
    return sigma
