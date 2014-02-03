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
        for val in range(len(self._classes)):
            self._classified_data.append([])
        for data_point in self._data:
            self._classified_data[self._classes.index(data_point[-1])].append(data_point)

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

    def kFoldCrossValidation(self, k):
        # Find the smallest class.
        size = None
        for data_class in self._classified_data:
            if size == None:
                size = len(data_class)
            elif size > len(data_class):
                size = len(data_class)

        # Determine ideal bucket size.
        bucket_size = size/(k)
        buckets = []
        for val in range(k):
            train = []
            test = []
            for cur_class in self.classes:
                train.append(np.array( \
                    self._classified_data[self.classes.index(cur_class)] \
                    [val*bucket_size:(val+1)*bucket_size]))

                left_side = self._classified_data \
                    [self.classes.index(cur_class)][0:val*bucket_size]
                right_side = self._classified_data \
                    [self.classes.index(cur_class)][(val+1)*bucket_size:]
                
                if len(left_side) > 0:
                    if len(test) == 0:
                        test = left_side
                    else:
                        test = np.append(test, left_side, axis=0)
                if len(right_side) > 0:
                    if len(test) == 0:
                        test = right_side
                    else:
                        test = np.append(test, right_side, axis=0)
            buckets.append([train, test])
        return buckets
                


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

    def __init__(self, classes, training_data, test_data):
        
        self.classes = classes
        self._test_data = test_data
        self._training_data = training_data
        self._cov_matrices = []
        self._det_cov_m = []
        self._inv_cov_m = []
        self._avg_training_data = []

        for cur_class in self.classes:
            class_index = self.classes.index(cur_class)
            current_training_data = []
            for val in self._training_data[class_index]:
                current_training_data.append(val[:-1])
            current_training_data = np.matrix(current_training_data)

            current_cov_matrix = np.cov(current_training_data.T)
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
        class_index = self.classes.index(class_index)
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
        for data_point in self._test_data:
            data_point = np.array(data_point)
            g_of_x = []
            for class_index in self.classes:
                g_of_x.append(self.__gOfX(class_index, data_point[:-1]))

            index_of_max = np.argmax(g_of_x)
            if self.classes[index_of_max] == int(data_point[-1]):
                correct += 1
            else:
                incorrect += 1
        return correct/(correct+incorrect)

class BayesianEstimation(object):
    """ Performs Bayesian Estimation on an input dataset.

    Attributes:
        training_data: Array of length of the classes, already split per class.
        test_data: Data to run the test on.
    """

    DET_EPSILON = 0.001

    def __init__(self, classes, training_data, test_data):
        self.classes = classes
        self._training_data = training_data
        self._cov_matrices = []
        self._det_cov_m = []
        self._inv_cov_m = []
        self._avg_training_data = []
        self._test_data = test_data

        for cur_class in self.classes:
            class_index = self.classes.index(cur_class)
            current_training_data = []
            for val in self._training_data[class_index]:
                current_training_data.append(val[:-1])

            current_training_data = np.matrix(current_training_data)
            current_cov_matrix = np.cov(current_training_data.T)

            self._cov_matrices.append(current_cov_matrix)
            self._det_cov_m.append(
                np.linalg.det(current_cov_matrix) + self.DET_EPSILON)

            # The next line was adapted from http://bit.ly/1iRa4Xj
            self._avg_training_data.append([sum(val)/len(val) for val in \
                itertools.izip(*current_training_data)])
        
        self._full_training_data = None
        for elem in self._training_data:
            if self._full_training_data == None:
                self._full_training_data = elem
            else:
                self._full_training_data = \
                    np.append(self._full_training_data, elem, axis=0)
        self._full_training_data = self._full_training_data[:,:-1]
        # The same line as above, adapted from http://bit.ly/1iRa4Xj
        self._avg_0 = np.matrix([sum(val)/len(val) for val in \
            itertools.izip(*(self._full_training_data))]).T
        self._cov_0 = np.cov(self._full_training_data.T)

        dist = (1.0/float(len(self._full_training_data)))

        for cur_class in self.classes:
            class_index = self.classes.index(cur_class)
            # Overlap in functions, only compute once.
            eq_left = self._cov_0 * np.linalg.inv(
                self._cov_0 + dist*self._cov_matrices[class_index])
            # Compute Bayesian mu_i.
            self._avg_training_data[class_index] = \
                np.matrix(self._avg_training_data[class_index][0]).T
            self._avg_training_data[class_index] = eq_left * \
                self._avg_training_data[class_index] + \
                dist*self._cov_matrices[class_index] * \
                np.linalg.inv(self._cov_0 + 
                    dist*self._cov_matrices[class_index]) * \
                self._avg_0

            # Compute Bayesian sigma_i.
            self._cov_matrices[class_index] = eq_left * dist * \
                self._cov_matrices[class_index]

            # Compute inverse.
            self._inv_cov_m.append(
                np.linalg.inv(self._cov_matrices[class_index]))

    def __gOfX(self, class_index, data_point):
        """ Private function to perform function g_i(x), where class_index = i.

            Attributes:
            class_index: Index of the class you want.
            num_samples: Number of samples you wish to train on.
        """
        # We convert this for datasets that classify starting at 1.
        class_index = self.classes.index(class_index)
        matrix_mult = \
            np.matrix(data_point - self._avg_training_data[class_index].T) * \
            np.matrix(self._inv_cov_m[class_index]) * \
            np.matrix(data_point - self._avg_training_data[class_index].T).T

        return -0.5*matrix_mult + math.log(0.5) - \
            (len(data_point)/2)*math.log(2*math.pi) - \
            0.5*math.log(self._det_cov_m[class_index])

    def run(self):
        correct = 0.0
        incorrect = 0.0
        # TODO(dcastro): Update this to test only on your test data.
        for data_point in self._test_data:
            data_point = np.array(data_point)
            g_of_x = []
            for class_index in self.classes:
                g_of_x.append(self.__gOfX(class_index, data_point[:-1]))
            
            index_of_max = np.argmax(g_of_x)
            if self.classes[index_of_max] == int(data_point[-1]):
                correct += 1
            else:
                incorrect += 1
        return correct/(correct+incorrect)


list_of_datasets = ["alcoholism_mod.csv",
                    "data_banknote_authentication.csv",
                    "Skin_NonSkin_mod.csv"]

for ds in list_of_datasets:
    cur_dataset = Dataset("../datasets/" + ds)
    setsforCV = cur_dataset.kFoldCrossValidation(10)

    cross_val_me = 0
    cross_val_be = 0
    for st in setsforCV:
        train = st[0]
        test = st[1]

        me = MaximumLikelihoodEstimation(cur_dataset.classes, train, test)
        cross_val_me += 100*me.run()

        be = BayesianEstimation(cur_dataset.classes, train, test)
        cross_val_be += 100*be.run()
    print "Processing: " + ds
    print "Cross Validation for Maximum Likelihood: " + str(cross_val_me/10)
    print "Cross Validation for Bayesian Estimation: " + str(cross_val_be/10)

    # Number of iterations
    ni = 30

    five_val_me = 0
    five_val_be = 0
    for val in range(ni):
        if ds == "alcoholism_mod.csv":
            five_samples = [cur_dataset.getTrainingData(7, 0),
                            cur_dataset.getTrainingData(7, 1)]
        else:
            five_samples = [cur_dataset.getTrainingData(5, 0),
                            cur_dataset.getTrainingData(5, 1)]
        me5 = MaximumLikelihoodEstimation(cur_dataset.classes, 
                                          five_samples,
                                          cur_dataset.data)
        be5 = BayesianEstimation(cur_dataset.classes, 
                                 five_samples,
                                 cur_dataset.data)
        five_val_me += 100*me5.run()
        five_val_be += 100*be5.run()

    print "Test for Maximum Likelihood - 5 samples:" + str(five_val_me/ni)
    print "Test for Bayesian Estimation - 5 samples:" + str(five_val_be/ni)

    fifty_val_me = 0
    fifty_val_be = 0
    for val in range(ni):
        fifty_samples = [cur_dataset.getTrainingData(50, 0),
                        cur_dataset.getTrainingData(50, 1)]
        me50 = MaximumLikelihoodEstimation(cur_dataset.classes, 
                                           fifty_samples,
                                           cur_dataset.data)
        be50 = BayesianEstimation(cur_dataset.classes,
                                  fifty_samples,
                                  cur_dataset.data)
        fifty_val_me += 100*me50.run()
        fifty_val_be += 100*be50.run()
    print "Test for Maximum Likelihood - 50 samples:" + str(fifty_val_me/ni)
    print "Test for Bayesian Estimation - 50 samples:" + str(fifty_val_be/ni)

