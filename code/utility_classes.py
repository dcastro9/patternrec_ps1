# Copyright 2014, All Rights Reserved
# Author: Daniel Castro <dcastro9@gatech.edu>

import csv
import itertools
import math
import numpy as np
import random

class Dataset(object):
    """ Represents a dataset. Can divide the dataset into 'k' classes,
    and provide training & testing data.

    Attributes:
       file_path: Location of the CSV data file.
    """

    def __init__(self, file_path):
        """Creates a Dataset object.
        """
        self._data = []
        self._classes = []
        self._classified_data = []
        with open(file_path, 'rb') as file_input:
            file_reader = csv.reader(file_input, delimiter=',')
            for data_point in file_reader:
                for val in range(len(data_point)):
                    data_point[val] = float(data_point[val])
                self._data.append(data_point)
                if int(data_point[-1]) not in self._classes:
                    self._classes.append(int(data_point[-1]))
        # Create an array for each class (if class start at 1, self._classes[0]
        # will be empty)
        # TODO(dcastro): There's a smarter way of doing this.
        for val in range(max(self._classes)+1):
            self._classified_data.append([])
        for data_point in self._data:
            self._classified_data[int(data_point[-1])].append(data_point[:-1])

    def getDataForClass(self, class_index):
        """ Obtains all the data for a given class.

        Attributes:
            class_index: Index of the class you want.
        """
        return np.array(self._classified_data[class_index])

    def getTrainingData(self, num_samples, class_index):
        """ Obtains a random sample of training data

        Attributes:
            num_samples: Number of random samples you want.
            class_index: Index of the class you want.
        """
        if num_samples > len(self._classified_data[class_index]):
            #TODO(dcastro): Revise if ValueError is the correct error to raise.
            raise ValueError("Not enough training samples.")
        return np.array(random.sample(
            self._classified_data[class_index], num_samples))

    @property
    def classes(self):
        return self._classes

    @property
    def data(self):
        return self._data

class MaximumLikelihoodEstimation(object):
    """ Performs the Maximum Likelihood Estimation on an input dataset.

    Attributes:
        file_path: Location of the CSV data file.
        num_samples: Number of samples you wish to train on.
    """

    DET_EPSILON = 0.001

    def __init__(self, file_path, num_samples):
        self._ds = Dataset(file_path)

        self._training_data = []
        self._cov_matrices = []
        self._det_cov_m = []
        self._inv_cov_m = []
        self._avg_training_data = []

        for cur_class in self._ds.classes:
            current_training_data = \
                self._ds.getTrainingData(num_samples, cur_class)
            current_cov_matrix = np.cov(current_training_data.T)

            self._training_data.append(current_training_data)
            self._cov_matrices.append(current_cov_matrix)
            self._det_cov_m.append(
                np.linalg.det(current_cov_matrix) + self.DET_EPSILON)
            self._inv_cov_m.append(np.linalg.inv(current_cov_matrix))
            # The next line was adapted from http://bit.ly/1iRa4Xj
            self._avg_training_data.append([sum(val)/len(val) for val in \
                itertools.izip(*current_training_data)])

    def __gOfX(self, class_index, data_point):
        """ Private function to perform function g_i(x), where class_index = i.

            Attributes:
            class_index: Index of the class you want.
            num_samples: Number of samples you wish to train on.
        """
        # We convert this for datasets that classify starting at 1.
        class_index = self._ds.classes.index(class_index)
        matrix_mult = \
            np.matrix(data_point - self._avg_training_data[class_index]) * \
            np.matrix(self._inv_cov_m[class_index]) * \
            np.matrix(data_point - self._avg_training_data[class_index]).T
        return -0.5*matrix_mult + math.log(0.5) - \
            (len(data_point)/2)*math.log(2*math.pi) - \
            0.5*math.log(self._det_cov_m[class_index])

    def run(self):
        correct = 0.0
        incorrect = 0.0
        # TODO(dcastro): Update this to test only on your test data.
        for data_point in self._ds.data:
            data_point = np.array(data_point)
            g_of_x = []
            for class_index in self._ds.classes:
                g_of_x.append(self.__gOfX(class_index, data_point[:-1]))

            index_of_max = np.argmax(g_of_x)
            if self._ds.classes[index_of_max] == int(data_point[-1]):
                correct += 1
            else:
                incorrect += 1
        return correct/(correct+incorrect)

mle = MaximumLikelihoodEstimation("../datasets/alcoholism.csv", 50)
print 100*mle.run()