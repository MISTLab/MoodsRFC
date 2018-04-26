import numpy as np
from Example_RF_classifier import build_features_vector
import pickle

from sklearn.metrics import accuracy_score


class DataLoader(object):
    """
    General utility class to load and build both the training dataset and the performance
    """
    def __init__(self):
        self._number_of_IMU_data_per_example = 50
        self._size_non_overlap = 5


    def format_data_to_train(self, vector_to_format_emg, vector_to_format_imu_1, vector_to_format_imu_2):
        """
        Function to format (calculate the features employed as input by the classifier and build the examples in their right shape)

        :param vector_to_format_emg: The emg recording to format into examples
        :param vector_to_format_imu_1: The IMU from the armband placed on the arm to format into examples
        :param vector_to_format_imu_2: The IMU from the armband placed on the leg to format into examples

        :return Return: Three arrays containing the examples for the EMG, IMU-Arm, IMU-Leg respectively
        """
        # We set number_of_IMU_data_per_example to 50 as a base so that each example is one second in length (IMU is cadenced at 50Hz)
        indice_EMG = 0
        indice_IMU_1 = 0
        indice_IMU_2 = 0
        example_imu_1 = []
        example_imu_2 = []
        example_emg = []

        dataset_examples_formatted_emg = []
        dataset_examples_formatted_imu_1 = []
        dataset_examples_formatted_imu_2 = []

        # Continue looping until one of the vector has run out of new data to give.
        while (indice_EMG < len(vector_to_format_emg) and indice_IMU_1 < len(vector_to_format_imu_1) and indice_IMU_2 < len(vector_to_format_imu_2)):
            if indice_EMG + 32 >= len(vector_to_format_emg) or indice_IMU_1 + 3 >= len(vector_to_format_imu_1) or indice_IMU_2 + 3 >= len(
                    vector_to_format_imu_2):  # We reached the end, we want to get free of the loop
                break

            for i in range(4):  # EMG is at 200Hz, IMU at 50 (4 EMG entry for 1 IMU entry)
                # There's eight EMG channels
                example_emg.append(vector_to_format_emg[indice_EMG:(indice_EMG + 8)])
                indice_EMG += 8
            # There's three IMU channels
            example_imu_1.append(vector_to_format_imu_1[indice_IMU_1:(indice_IMU_1 + 3)])
            example_imu_2.append(vector_to_format_imu_2[indice_IMU_2:(indice_IMU_2 + 3)])
            indice_IMU_1 += 3
            indice_IMU_2 += 3
            if len(example_imu_1) >= self._number_of_IMU_data_per_example:
                # Reshape the example to have 10 sub-examples per examples
                example_emg = np.reshape(example_emg, newshape=(10, 20, 8)).tolist()
                example_imu_1 = np.reshape(example_imu_1, newshape=(10, 5, 3)).tolist()
                example_imu_2 = np.reshape(example_imu_2, newshape=(10, 5, 3)).tolist()

                # Get the features for the EMG and IMU
                emg_example, imu_1_example, imu_2_example = build_features_vector.build_features_vector(example_emg,
                                                                                                        example_imu_1,
                                                                                                        example_imu_2)

                dataset_examples_formatted_emg.append(emg_example)
                dataset_examples_formatted_imu_1.append(imu_1_example)
                dataset_examples_formatted_imu_2.append(imu_2_example)

                # Reshape the arrays the way they were constructed
                example_emg = np.reshape(example_emg, newshape=(200, 8)).tolist()
                example_imu_1 = np.reshape(example_imu_1, newshape=(50, 3)).tolist()
                example_imu_2 = np.reshape(example_imu_2, newshape=(50, 3)).tolist()

                # Remove only part of the data accumulated to obtain sliding window over the dataset
                example_emg = example_emg[4 * self._size_non_overlap::]
                example_imu_1 = example_imu_1[self._size_non_overlap::]
                example_imu_2 = example_imu_2[self._size_non_overlap::]

        return np.array(dataset_examples_formatted_emg), np.array(dataset_examples_formatted_imu_1), np.array(dataset_examples_formatted_imu_2)

    def read_training(self, path, number_of_classes):
        """
        Function to build the training dataset.

        :param path: the path that contain the training dataset recording
        :param number_of_classes: number of moods made by the performer during training

        :return Return: Four arrays containing the training dataset for the EMG, IMU-Arm, IMU-Leg recording and the labels respectively
        """
        try:
            print("Reading Data")
            X_emg = []
            X_imu_1 = []
            X_imu_2 = []
            Y = []
            print(number_of_classes)
            for i in range(number_of_classes * 3):
                data_read_from_file_emg = np.fromfile(path + "\\classe_%d_emg.dat" % i, dtype=np.int32)
                data_read_from_file_imu_1 = np.fromfile(path + "\\classe_%d_first_imu.dat" % i,
                                                        dtype=np.float32)
                data_read_from_file_imu_2 = np.fromfile(path + "\\classe_%d_second_imu.dat" % i,
                                                        dtype=np.float32)

                dataset_examples_formatted_emg, dataset_examples_formatted_imu_1, dataset_examples_formatted_imu_2 = self.format_data_to_train(
                    data_read_from_file_emg, data_read_from_file_imu_1, data_read_from_file_imu_2)

                X_emg.extend(dataset_examples_formatted_emg)
                X_imu_1.extend(dataset_examples_formatted_imu_1)
                X_imu_2.extend(dataset_examples_formatted_imu_2)
                if i < number_of_classes:
                    Y.extend(i + np.zeros(dataset_examples_formatted_imu_1.shape[0]))
                elif i < number_of_classes * 2:
                    Y.extend((i - number_of_classes) + np.zeros(dataset_examples_formatted_imu_1.shape[0]))
                else:
                    Y.extend((i - (number_of_classes * 2)) + np.zeros(dataset_examples_formatted_imu_1.shape[0]))
            print(Y)

            return X_emg, X_imu_1, X_imu_2, Y
        except Exception as e:
            print(e)

    def read_performance_data(self, path_seance, path_labels, dataset_delay):
        """
        Function to build the training dataset.

        :param path_seance: the path that contain the performance recording
        :param path_labels: the path that contain the true labels for this performance
        :param dataset_delay: the number of second that separate the recording from the Armbands with the video utilized to classify the dataset

        :return Return: Five arrays containing the performance dataset for the EMG, IMU-Arm, IMU-Leg recording. the labels and the accuracy
        obtained by the
        classifier during the live performance respectively
        """
        dataset = pickle.load(open(path_seance + "\\dataset_predicted.p", "rb"))
        true_labels_array = np.array(path_labels)

        prediction = []
        dataset_emg = []
        dataset_imu_1 = []
        dataset_imu_2 = []
        timestamp = []
        all_emg = []
        all_imu_1 = []
        all_imu_2 = []

        index_timestamp_delay = -1
        index = 0

        current_index_true_label = 0
        true_labels_seance = []
        '''
        j will be an array containing all the information relating to the current example within the performance dataset.
        j[0] contains the EMG recording of the example
        j[1] contains the IMU recording of the Myo on the arm
        j[2] contains the IMU recording of the Myo on the leg
        j[3] contains the timestamp of every examples
        j[4] contains the live prediction of the current example obtained during the performance
        '''
        for j in dataset:
            current_timestamp = j[3]

            # If the current time of the example is over the end of the performance, stop reading the performance data
            if true_labels_array[current_index_true_label][0] == -1 and \
                    current_timestamp >= true_labels_array[current_index_true_label][1]:
                break

            # If the current time of the example is over the next true label timestamp, change update the true label employed.
            if current_timestamp >= true_labels_array[0][1]:
                if current_timestamp >= true_labels_array[current_index_true_label][1]:
                    current_index_true_label += 1
                true_labels_seance.append(true_labels_array[current_index_true_label - 1][0])
            timestamp.append(current_timestamp)

            if index_timestamp_delay == -1 and current_timestamp >= dataset_delay:
                index_timestamp_delay = index

            # Build the performance dataset while synchronizing it to the video. The synchronization was made manually using the visual cues from the
            # swarm activity.
            if dataset_delay < 0:
                if current_timestamp > -1 * dataset_delay:
                    emg = np.array(j[0]).reshape(len(j[0]) * len(j[0][0]), len(j[0][0][0]))
                    all_emg.append(emg)
                    imu_1 = np.array(j[1]).reshape(len(j[1]) * len(j[1][0]), len(j[1][0][0]))
                    all_imu_1.append(imu_1)
                    imu_2 = np.array(j[2]).reshape(len(j[2]) * len(j[2][0]), len(j[2][0][0]))
                    all_imu_2.append(imu_2)

                    emg_example, imu_1_example, imu_2_example = build_features_vector.build_features_vector(j[0], j[1], j[2])

                    dataset_emg.append(emg_example)
                    dataset_imu_1.append(imu_1_example)
                    dataset_imu_2.append(imu_2_example)

                    prediction.append(int(j[4]))
            else:
                emg = np.array(j[0]).reshape(len(j[0]) * len(j[0][0]), len(j[0][0][0]))
                all_emg.append(emg)
                imu_1 = np.array(j[1]).reshape(len(j[1]) * len(j[1][0]), len(j[1][0][0]))
                all_imu_1.append(imu_1)
                imu_2 = np.array(j[2]).reshape(len(j[2]) * len(j[2][0]), len(j[2][0][0]))
                all_imu_2.append(imu_2)

                emg_example, imu_1_example, imu_2_example = build_features_vector.build_features_vector(j[0], j[1], j[2])

                dataset_emg.append(emg_example)
                dataset_imu_1.append(imu_1_example)
                dataset_imu_2.append(imu_2_example)

                prediction.append(int(j[4]))
            index += 1

        # Synchronize the recording of the armbands of with the true labels obtained by watching the performance's video.
        if dataset_delay > 0:
            true_labels_seance = true_labels_seance[index_timestamp_delay::]

        dataset_emg = dataset_emg[:len(true_labels_seance)]
        dataset_imu_1 = dataset_imu_1[:len(true_labels_seance)]
        dataset_imu_2 = dataset_imu_2[:len(true_labels_seance)]

        print("EMG : ", np.shape(dataset_emg))
        print("IMU 1 : ", np.shape(dataset_imu_1))
        print("IMU 2 : ", np.shape(dataset_imu_2))

        prediction = prediction[:len(true_labels_seance)]
        print("PREDICTION : ", prediction)
        print("TRUE LABELS: ", true_labels_seance)
        print("ACCURACY PREDICTIONS: ", accuracy_score(true_labels_seance, prediction))
        accuracy = accuracy_score(true_labels_seance, prediction)

        return dataset_emg, dataset_imu_1, dataset_imu_2, true_labels_seance, accuracy