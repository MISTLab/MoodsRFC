import numpy as np
from Example_RF_classifier import extract_features
__author__ = "Ulysse Cote-Allard<ulysse.cote-allard.1@ulaval.ca> and David St-Onge<david.st-onge@polymtl.ca>"
__copyright__ = "Copyright 2007, MIST Lab"
__credits__ = ["David St-Onge", "Ulysse Cote-Allard", "Kyrre Glette", "Benoit Gosselin", "Giovanni Beltrame"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "David St-Onge"
__email__ = "david.st-onge@polymtl.ca"
__status__ = "Production"

def build_features_vector(list_example_emg, list_example_imu_1, list_example_imu_2):
    """

    :param list_example_emg: The EMG array to calculate the examples from using the features described in the article
    :param list_example_imu_1: The IMU from the armband placed on the arm to calculate the examples from using the features described in the article
    :param list_example_imu_2: The IMU from the armband placed on the leg to calculate the examples from using the features described in the article

    :return Return: Three arrays containing the examples for the EMG, IMU-Arm, IMU-Leg respectively
    """
    # Build features vector for EMG
    list_features_emg = []
    for example in list_example_emg:
        for i in range(0, np.shape(example)[1]):
            example = np.array(example)
            vector_canal = example[:, i]
            if i in [0, 4]:
                list_features_emg.append(np.var(vector_canal))
                list_features_emg.append(extract_features.iemg(vector_canal))
                list_features_emg.append(extract_features.mav(vector_canal))
                list_features_emg.append(extract_features.rms(vector_canal))

    # Build feature vector for IMU on the arm
    list_features_imu_1 = []
    for example in list_example_imu_1:
        for i in range(0, np.shape(example)[1]):
            example = np.array(example)
            vector_canal = example[:, i]
            if i is not 2:
                list_features_imu_1.append(np.max(vector_canal))
                list_features_imu_1.append(np.mean(vector_canal))
                list_features_imu_1.append(extract_features.iemg(vector_canal))
                list_features_imu_1.append(extract_features.mav(vector_canal))
                list_features_imu_1.append(extract_features.rms(vector_canal))
            list_features_imu_1.append(np.var(vector_canal))

    # Build feature vector for IMU on the leg
    list_features_imu_2 = []
    for example in list_example_imu_2:
        for i in range(0, np.shape(example)[1]):
            example = np.array(example)
            vector_canal = example[:, i]
            if i is not 2:
                list_features_imu_2.append(np.max(vector_canal))
                list_features_imu_2.append(np.mean(vector_canal))
                list_features_imu_2.append(extract_features.iemg(vector_canal))
                list_features_imu_2.append(extract_features.mav(vector_canal))
                list_features_imu_2.append(extract_features.rms(vector_canal))
            list_features_imu_2.append(np.var(vector_canal))

    return np.array(list_features_emg), np.array(list_features_imu_1), np.array(list_features_imu_2)
