
import numpy as np
import tensorflow as tf

class MLP:

    def __init__(self, options):
        """Constructor initializes the parameters of the MLP"""

        # size layers states the number of neurons in each layer,
        # starting from the input layer all the way to the output layer
        self.__sizeLayers = options["sizeLayers"]
        self.__maxEpochs = options["maxEpochs"]
        self.__learningRate = options["learningRate"]
        self.__minCostDiff = options["minCostDiff"]

    def trainMLP(self, trainInputs, trainTargets):
        """Trains the MLP.
        trainInputs is assumed to be of size [numFeatures, numSamples] and trainTargets be of size [numClasses, numSamples]"""

        x = tf.placeholder(trainInputs)
        y = tf.placeholder(trainTargets)

        # define the weight matrices between each layer
        # CAUTION: weight matrices already contain bias terms as trainInputs contains 1 in the last row
        weight_matrix_list = []
        for i in range(len(self.__sizeLayers) - 1):
            weight_matrix_list.append(tf.Variable(tf.random_normal([self.__sizeLayers[i+1], self.__sizeLayers[i]], mean = 0, stddev = 1)))

        # we should store the outputs of each layer in the feed-forward phase
        # for backpropagation later
        outputs_list = []
        outputs_list.append(x)

        # feed-forward phase
        # feed-forward until (excluding) the softmax layer
        for i in range(len(self.__sizeLayers) - 2):
            outputs_list.append(tf.sigmoid(tf.matmul(weight_matrix_list[i], outputs_list[i])))

        # output of the softmax layer
        z = tf.sigmoid(tf.matmul(weight_matrix_list[len(self.__sizeLayers) - 2], outputs_list[len(self.__sizeLayers) - 2]))
        zSum = tf.tile(tf.cumsum(z, axis = 0), tf.constant([train_targets.shape[0], 1]))
        p = tf.div(z, zSum)
        outputs_list.append(p)

        # cross-entropy cost
        cost = -tf.cumsum(tf.multiply(tf.log(p), trainTargets), axis = 0)

        # backprop phase
        
        

        

        
