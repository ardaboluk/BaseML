
import numpy as np
import tensorflow as tf

class MLP:

    def __init__(self, options):
        """Constructor initializes the parameters of the MLP"""

        # size layers states the number of neurons in each layer starting from the input layer all the way to the output layer
        # dropouts contains the keep probabilities for each layer (including the input layer)
        self.__sizeLayers = options["sizeLayers"]
        self.__use_dropouts = options["use_dropouts"]
        self.__dropouts = options["dropouts"]
        self.__maxEpochs = options["maxEpochs"]
        self.__learningRate = options["learningRate"]
        self.__minCostDiff = options["minCostDiff"]
        self.__numClasses = None

        self.__sess = None
        self.__weight_matrix_list = []

    def trainMLP(self, trainInputs, trainTargets):
        """Trains the MLP.
        trainInputs is assumed to be of size [numFeatures, numSamples] and trainTargets be of size [numClasses, numSamples]"""

        self.__numClasses = trainTargets.shape[0]
        
        # None means size can be anything
        x = tf.placeholder(tf.float32, [trainInputs.shape[0], None])
        y = tf.placeholder(tf.float32, [self.__numClasses, None])

        # define the weight matrices between each layer
        # CAUTION: weight matrices already contain bias terms as trainInputs contains 1 in the last row
        self.__weight_matrix_list = []
        for i in range(len(self.__sizeLayers) - 1):
            self.__weight_matrix_list.append(tf.Variable(tf.random_uniform(shape = [self.__sizeLayers[i+1], self.__sizeLayers[i]], minval = -(1-1e-8), maxval = 1)))

        # define inverted dropout masks
        # we shouldn't assign these to tf.Variables as doing this would keep the old values while we want different ones after each run()
        dropout_masks_list = []
        for i in range(len(self.__sizeLayers) - 1):
            rand_values = tf.random_uniform(shape = [self.__sizeLayers[i], trainInputs.shape[1]], minval = -(1-1e-8), maxval = 1)
            dropout_masks_list.append(tf.div(tf.to_float(tf.less(rand_values, self.__dropouts[i])), self.__dropouts[i]))

        # feed-forward until (excluding) the softmax layer and apply dropout along the way
        # we don't have to store the outputs of each layer in the feed-forward phase
        # as we'll use tensorflow to calculate the gradients
        cur_layer_output = x
        cur_layer_output = tf.multiply(cur_layer_output, dropout_masks_list[0])
        for i in range(len(self.__sizeLayers) - 2):
            cur_layer_output = tf.sigmoid(tf.matmul(self.__weight_matrix_list[i], cur_layer_output))
            cur_layer_output = tf.multiply(cur_layer_output, dropout_masks_list[i+1])

        # output of the softmax layer
        z = tf.sigmoid(tf.matmul(self.__weight_matrix_list[len(self.__sizeLayers) - 2], cur_layer_output))
        zSum = tf.tile(tf.reduce_sum(z, axis = 0, keep_dims = True), tf.constant([self.__numClasses, 1]))
        predictions = tf.div(z, zSum)

        # cross-entropy cost
        cost = - tf.reduce_sum(tf.multiply(tf.log(predictions), y))

        # backprop phase
        opt = tf.train.GradientDescentOptimizer(self.__learningRate)
        grads_and_variables = opt.compute_gradients(cost, self.__weight_matrix_list)
        train_step = opt.apply_gradients(grads_and_variables)

        # initialize variables
        init = tf.global_variables_initializer()

        # train the model
        prev_cost = 0
        cur_cost = prev_cost
        self.__sess = tf.Session()
        self.__sess.run(init)        
        for i in range(self.__maxEpochs):

            # here, we shouldn't call run to get the cost, that would do the forward-pass again, which is an overhead
            _, cur_cost = self.__sess.run([train_step, cost], feed_dict = {x:trainInputs, y:trainTargets})
            cur_cost_diff = abs(cur_cost - prev_cost)
            prev_cost = cur_cost
            
            print("Epoch {} Cost {} Cost Diff {}".format(i+1, cur_cost, cur_cost_diff))

            if  cur_cost_diff <= self.__minCostDiff:
                print("In Epoch {} cost difference {} is under pre-defined min-cost {}, stopping training..".format(i+1, cur_cost_diff, self.__minCostDiff))
                break


    def predict(self, testInputs):
        """Returns predictions for the given test inputs."""

        x = tf.placeholder(tf.float32, [testInputs.shape[0], None])

        cur_layer_output = x
        for i in range(len(self.__sizeLayers) - 2):
            cur_layer_output = tf.sigmoid(tf.matmul(self.__weight_matrix_list[i], cur_layer_output))

        z = tf.sigmoid(tf.matmul(self.__weight_matrix_list[len(self.__sizeLayers) - 2], cur_layer_output))
        zSum = tf.tile(tf.reduce_sum(z, axis = 0, keep_dims = True), tf.constant([self.__numClasses, 1]))
        predictions = tf.div(z, zSum)

        preds = self.__sess.run(predictions, feed_dict = {x:testInputs})
        
        return preds
        
        
