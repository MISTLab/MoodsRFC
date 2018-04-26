import numpy as np
import math
__author__ = "Ulysse Cote-Allard<ulysse.cote-allard.1@ulaval.ca> and David St-Onge<david.st-onge@polymtl.ca>"
__copyright__ = "Copyright 2007, MIST Lab"
__credits__ = ["David St-Onge", "Ulysse Cote-Allard", "Kyrre Glette", "Benoit Gosselin", "Giovanni Beltrame"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "David St-Onge"
__email__ = "david.st-onge@polymtl.ca"
__status__ = "Production"

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
