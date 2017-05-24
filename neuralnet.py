"""
This module will provide a factory to create and
train Neural Networks with TensorFlow
"""

import numpy as np
import trainer
import tensorflow as tf


class NNLayer(object):
    """Trainable Neural Network Layer"""

    def __init__(self, _input_connections, _neurons, _layer_type='hidden'):
        """
        Initialize the Neural Layer

        Args:
            _input_size: size of the layer input (number of in connections)
            _neurons: number of layer neurons
            _layer_type: 'input' , 'hidden' , 'output'
        """

        # Create the trainable values

        # Input Data placeholder
        if _layer_type is 'input':
            self.x = tf.placeholder(tf.float32, [_input_connections[0], _input_connections[1]])
            self.W = tf.Variable(tf.truncated_normal([_input_connections[1], _neurons], stddev=1) / tf.sqrt(2.0 * _input_connections[1]), tf.float32)
            self.b = tf.Variable(tf.truncated_normal([1, _neurons], stddev=0.01), tf.float32)
        else:
            self.x = _input_connections
            self.W = tf.Variable(tf.truncated_normal([_input_connections.shape[1].value, _neurons], stddev=1)  * tf.sqrt(2.0 / _input_connections.shape[1].value), tf.float32)
            self.b = tf.Variable(tf.truncated_normal([1, _neurons], stddev=0.01), tf.float32)

        # Weights and output calculation
        if _layer_type is not 'output':
            self.computation = tf.nn.relu(tf.add(tf.matmul(self.x, self.W), self.b))
        else:
            self.computation = tf.add(tf.matmul(self.x, self.W), self.b)



class NeuralNetwork(trainer.Trainer):
    """Trainable Neural Network"""

    def __init__(self, _structure):
        """
        Initialize the Neural Network

        Args:
            _structure: Shape of the network layers
                        Each value in the List shall describe the
                        number of neurons in that layer
                        [Input, Hidden_1, ... , Hidden_N, Output]
                        Input layer shall have at least 1 input
                        Output layer shall have at least 1 output

        Raise:  RuntimeError if the passed structure is not compliant.
        """

        # Runtime error if the network structure is not compliant
        if not isinstance(_structure, list) or len(_structure) < 2 or _structure[0] <= 0 or _structure[-1] <= 0:
            raise RuntimeError("Bad Neural Network structure")

        # Create the Generic Trainer
        super().__init__()

        # Create the network layers
        self.network_layers = []
        self.network_layers.append(NNLayer([None, _structure[0]], _structure[1], _layer_type='input'))
        for layer in range(2, len(_structure)):
            if layer < (len(_structure) - 1):
                self.network_layers.append(NNLayer(self.network_layers[-1].computation, _structure[layer], _layer_type='hidden'))
            else:
                self.network_layers.append(NNLayer(self.network_layers[-1].computation, _structure[layer], _layer_type='output'))

        # Create the prediction and trainable variables
        self.prediction = tf.nn.softmax(self.network_layers[-1].computation)


    def ConfigureTraining(self, _learning_rate, _decay_steps, _decay_amount, _min_epochs_without_progress=0):
        """
        Configure the Neural Network training

        Args:
            _learning_rate:  Staring Learning Rate
            _decay_steps: amount of steps to apply a decay
            _decay_amount: percentage of the previous learning rate to mantain ageter decay steps
            _min_epochs_without_progress: Minimum number of epochs without progress to run
        """
        self.min_epochs_without_progress = _min_epochs_without_progress

        # One-hot encoded truth value placeholder
        self.truth = tf.placeholder(tf.float32, [None, 10])

        # Loss calculation
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.truth * tf.log(self.prediction)))

        # Learning Step Decay
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.learning_rate = tf.train.exponential_decay(_learning_rate, self.global_step, _decay_steps, _decay_amount, staircase=False)

        # Train node
        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy, global_step=self.global_step)

        # validation operations
        self.val_predictions = tf.placeholder(tf.float32, [None, 10])
        self.val_labels = tf.placeholder(tf.float32, [None, 10])
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.val_predictions, 1), tf.arg_max(self.val_labels, 1)), tf.float32))


    def RunTrainingStep(self, _input_data):
        """
        Exeute a training step of the network

        Args:
            _input_data: input data for the input layer

        Returns: loss value
        """

        data = []
        label = []
        for batch in _input_data:
            data.append(batch['data'])
            label.append(batch['label'])

        pred, loss, _ = (self.tf_session.run([self.prediction, self.cross_entropy, self.train],
                                             feed_dict={self.network_layers[0].x: data, self.truth: label}))
        
        self.AddTrainingResult(loss, len(_input_data))

        return loss


    def RunPrediction(self, _input_data, _batch_size):
        """
        Exeute a prediction step of the network

        Args:
            _input_data: input data for the input layer
            _batch_size: number of data to process in each iteration

        Returns: prediction accuracy
        """
        batch_available_num = len(_input_data)
        batch_number = 0
        accuracy = 0
        while batch_available_num > 0:

            data = []
            label = []

            retrieve_num = min(_batch_size, batch_available_num)
            start = len(_input_data) - batch_available_num
            stop = start + retrieve_num
            batch_available_num -= retrieve_num

            # Add current element in the batch
            for element in _input_data[start : stop]:
                data.append(element['data'])
                label.append(element['label'])

            predictions = self.tf_session.run([self.prediction], feed_dict={self.network_layers[0].x: data})
            accuracy += self.tf_session.run(self.accuracy, feed_dict={self.val_predictions: predictions[0], self.val_labels: label})
            batch_number += 1

        return accuracy / batch_number


    def AddTestResults(self, _train_dset, _test_dset, _batch_size):
        """
        Add the results for a test run

        Args:
            _train_dset: training dataset
            _test_dset: test dataset
            _batch_size: Batch size for testing

        Returns:
            True if the results are the best ones, False otherwise
        """
        _train_accuracy = self.RunPrediction(_train_dset, _batch_size)
        _test_accuracy = self.RunPrediction(_test_dset, _batch_size)

        is_best_epoch = self.RecordAccuracy(_train_accuracy, _test_accuracy)
        if is_best_epoch:
            for layer in self.network_layers:
                if hasattr(layer, 'W'):
                    self.best_weights.append(self.tf_session.run(layer.W))

        return is_best_epoch


    def ProcessWeightsForDisplay(self, _weights, _width, _height):
        """
        Setup the weights to be displayed

        Args:
            _weights: weights to display
            _width: display image width
            _height: display image height
        """

        W = np.transpose(_weights)
        return np.reshape(W, [W.shape[0], _width, _height])


    def DisplayWeights(self, _weights, _width, _height, _plot_rows, _plot_cols):
        """
        Draw the passed training weights, each column is a different "image" filter
        learned

        Args:
            _weights: weights to display in displayable format [batch, [width, height]]
            _width: display image width
            _height: display image height
            _plot_rows: rows to arrange the plotted data
            _plot_cols: columns to arrange the plotted data
        """

        # Preprocess weights to be ready for display
        weights = self.ProcessWeightsForDisplay(_weights, _width, _height)

        # Display weights
        self.DisplayPlot(weights, _plot_rows, _plot_cols)




