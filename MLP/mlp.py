
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

        self.__sess = None

    def trainMLP(self, trainInputs, trainTargets):
        """Trains the MLP.
        trainInputs is assumed to be of size [numFeatures, numSamples] and trainTargets be of size [numClasses, numSamples]"""

        x = tf.placeholder(tf.float32, [trainInputs.shape[0], None], name = "datainput")
        y = tf.placeholder(tf.float32, [trainTargets.shape[0], None])

        # define the weight matrices between each layer
        # CAUTION: weight matrices already contain bias terms as trainInputs contains 1 in the last row
        weight_matrix_list = []
        for i in range(len(self.__sizeLayers) - 1):
            weight_matrix_list.append(tf.Variable(tf.random_normal([self.__sizeLayers[i+1], self.__sizeLayers[i]], mean = 0, stddev = 1)))

        # feed-forward until (excluding) the softmax layer
        # we don't have to store the outputs of each layer in the feed-forward phase
        # as we'll use tensorflow to calculate the gradients
        cur_layer_output = x
        for i in range(len(self.__sizeLayers) - 2):
            cur_layer_output = tf.sigmoid(tf.matmul(weight_matrix_list[i], cur_layer_output))

        # output of the softmax layer
        z = tf.sigmoid(tf.matmul(weight_matrix_list[len(self.__sizeLayers) - 2], cur_layer_output))
        zSum = tf.tile(tf.reduce_sum(z, axis = 0, keep_dims = True), tf.constant([trainTargets.shape[0], 1]))
        predictions = tf.div(z, zSum, name = "predictions")

        # cross-entropy cost
        cost = - tf.reduce_sum(tf.multiply(tf.log(predictions), y))

        # backprop phase
        opt = tf.train.GradientDescentOptimizer(self.__learningRate)
        grads_and_variables = opt.compute_gradients(cost, weight_matrix_list)
        train_step = opt.apply_gradients(grads_and_variables)

        # initialize variables
        init = tf.global_variables_initializer()

        # train the model
        prev_cost = 0
        cur_cost = prev_cost
        self.__sess = tf.Session()
        self.__sess.run(init)        
        for i in range(self.__maxEpochs):
            
            self.__sess.run(train_step, feed_dict = {x:trainInputs, y:trainTargets})
            cur_cost = self.__sess.run(cost, feed_dict = {x:trainInputs, y:trainTargets})
            
            print("Epoch {} Cost {}".format(i+1, cur_cost))
            
            cur_cost_diff = abs(cur_cost - prev_cost)
            prev_cost = cur_cost

            if  cur_cost_diff <= self.__minCostDiff:
                print("In Epoch {} cost difference {} is under pre-defined min-cost {}, stopping training..".format(i+1, cur_cost_diff, self.__minCostDiff))
                break


    def predict(self, testInputs):
        """Returns predictions for the given test inputs."""

        predictions = self.__sess.graph.get_tensor_by_name("predictions:0")
        preds = self.__sess.run(predictions, feed_dict = {"datainput:0":testInputs})
        return preds
        
        
