import numpy as np
from os import listdir
from Example_RF_classifier import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
__author__ = "Ulysse Cote-Allard<ulysse.cote-allard.1@ulaval.ca> and David St-Onge<david.st-onge@polymtl.ca>"
__copyright__ = "Copyright 2007, MIST Lab"
__credits__ = ["David St-Onge", "Ulysse Cote-Allard", "Kyrre Glette", "Benoit Gosselin", "Giovanni Beltrame"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "David St-Onge"
__email__ = "david.st-onge@polymtl.ca"
__status__ = "Production"

if __name__ == '__main__':
    """
    Main function utilized to build the training dataset and the performance of each participant and then training and using the classifier
    """
    dataLoader = load_data.DataLoader()
    # Get the list of participants from the true labels folder
    list_participants = listdir("../true_labels_informations/")

    accuracy_RF_array = []
    accuracy_CNN_array = []
    number_gestures_array = []

    accuracy_calculated_RF = []

    array_participants_id = []

    examples_imu_training_datasets = []
    examples_emg_training_datasets = []
    labels_training_datasets = []
    examples_imu_test_datasets = []
    examples_emg_test_datasets = []
    labels_test_datasets = []
    for participant in list_participants:
        identification_participant = participant.split(".")[0]
        information_labels = np.load("../true_labels_informations/" + participant, encoding="bytes")
        dataset_delay, labels_intervals, number_gestures, true_labels = information_labels
        number_gestures_array.append(number_gestures[0])
        print(identification_participant)
        array_participants_id.append(identification_participant)

        # Comment between here

        test_data_emg, test_data_imu_1, test_data_imu_2, labels_test, accuracy_RF = dataLoader.read_performance_data("../" +
                                                                                                                     identification_participant,
                                                                                                                     path_labels=labels_intervals,
                                                                                                                     dataset_delay=dataset_delay[0])

        np.savez("../formated_datasets/" + identification_participant + "_accuracy_RF.npz", accuracy_RF=[accuracy_RF])
        examples_emg, examples_imu_1, examples_imu_2, labels = dataLoader.read_training(
            "../" + identification_participant,
            number_of_classes=number_gestures[0])

        np.savez("../formated_datasets/" + identification_participant + "_train_rf_features.npz", examples_emg=examples_emg,
                 examples_imu_1=examples_imu_1, examples_imu_2=examples_imu_2, labels=labels)

        np.savez("../formated_datasets/" + identification_participant + "_test_rf_features.npz", test_data_emg=test_data_emg,
                 test_data_imu_1=test_data_imu_1, test_data_imu_2=test_data_imu_2, labels_test=labels_test)

        # And here to stop building the datasets from scratch

        npzfile = np.load("../formated_datasets/" + identification_participant + "_accuracy_RF.npz", encoding="bytes")
        accuracy_RF_live_performance = npzfile['accuracy_RF'][0]

        npzfile = np.load("../formated_datasets/" + identification_participant + "_train_rf_features.npz", encoding="bytes")
        emg_train, examples_imu_1_pre_training, examples_imu_2_pre_training, labels_pre_training = npzfile['examples_emg'],\
                                                                                                   npzfile['examples_imu_1'],\
                                                                                                   npzfile['examples_imu_2'], npzfile['labels']

        npzfile = np.load("../formated_datasets/" + identification_participant + "_test_rf_features.npz", encoding="bytes")
        emg_test, test_data_imu_1, test_data_imu_2, labels_test = npzfile['test_data_emg'], npzfile['test_data_imu_1'], npzfile['test_data_imu_2'],\
                                                                  npzfile['labels_test']

        rf = RandomForestClassifier(n_estimators=500, max_features='log2', random_state=np.random.RandomState(42))
        rf.fit(np.concatenate((emg_train, examples_imu_1_pre_training, examples_imu_2_pre_training), axis=1), labels_pre_training)
        predictions = rf.predict(np.concatenate((emg_test, test_data_imu_1, test_data_imu_2), axis=1))

        accuracy_RF = accuracy_score(labels_test, predictions)

        print("ACCURACY RF LIVE PERFORMANCE: ", accuracy_RF_live_performance)
        print("ACCURACY RF CALCULATED: ", accuracy_RF)
        print("CONFUSION MATRIX: \n", confusion_matrix(labels_test, predictions))
        print("  ")
        print(npzfile.files)
        accuracy_calculated_RF.append(accuracy_RF)

    print("ACCURACY RF FROM ALL DATASETS:", accuracy_calculated_RF)
    print("participant ID: ", array_participants_id)
