"""
This module will provide a factory to create and
train Convolutional Networks with TensorFlow
"""

import numpy as np
import trainer
import tensorflow as tf


class ConvLayer(object):
    """Trainable Convolution Layer"""

    def __init__(self, _input_data, _input_shape, _kernel_shape, _stride_shape, _is_input_layer=False):
        """
        Initialize the Convolutional Layer

        Args:
            _input_data: Input tensor formatted as [batch, (width * height * channels) pixels]
            _input_shape: a tensor with the input shape [width,height,channels]
            _kernel_shape: a tensor with the kernel shape [width, height, channels, deepness]
            _stride_shape: a tensor with the stride shape [horizontal, vertical]
            _is_input_layer: if True, the input values are provided by the user and not
            inherited from the previous layers
        """

        # Check input correctness
        if len(_input_shape) is not 3:
            raise RuntimeError("Input shape should be [width, height, channels]")
        if len(_kernel_shape) is not 4:
            raise RuntimeError("Kernel shape should be [width, height, channels, deepness]")
        if len(_stride_shape) is not 2:
            raise RuntimeError("Stride shape should be [horizontal, vertical]")

        # Input
        if not _is_input_layer:
            # Previous level input
            self.x = _input_data
        else:
            # Reshape Input -> # Batch/Height/Width/Channels
            self.input = tf.placeholder(tf.float32, [None, (_input_shape[0] *_input_shape[1] * _input_shape[2])])
            self.x = tf.reshape(self.input, [-1, _input_shape[1], _input_shape[0], _input_shape[2]])

        # Weights -> Kernel Height/Width/Channels/Deepness
        self.W = tf.Variable(tf.truncated_normal([_kernel_shape[1], _kernel_shape[0], _kernel_shape[2], _kernel_shape[3]], stddev=1) * tf.sqrt(2.0 / (_kernel_shape[1] * _kernel_shape[0] * _kernel_shape[2])), tf.float32)

        # Bias -> Deepness
        self.b = tf.Variable(tf.truncated_normal([_kernel_shape[3]], stddev=0.1), tf.float32)

        # Perform the convolution
        self.computation = tf.nn.relu(tf.nn.conv2d(self.x, self.W, strides=[1, _stride_shape[1], _stride_shape[0], 1], padding='SAME') + self.b)



class PoolingLayer(object):
    """Pooling Layer"""

    def __init__(self, _input_data, _pool_shape, _stride_shape):
        """
        Initialize the Pooling Layer

        Args:
            _input_data: input data to calculate the pooling on
            _pool_shape: Shape of the pooling kernel [width, height]
            _stride_shape: a tensor with the stride shape [horizontal, vertical]
        """

        # Check input correctness
        if len(_pool_shape) is not 2:
            raise RuntimeError("Pool kernel shape should be [width, height]")
        if len(_stride_shape) is not 2:
            raise RuntimeError("Stride shape should be [horizontal, vertical]")

        # Pooling operation
        self.computation = tf.nn.max_pool(_input_data, ksize=[1, _pool_shape[1], _pool_shape[0], 1], strides=[1, _stride_shape[1], _stride_shape[0], 1], padding='SAME')



class FCLayer(object):
    """Fully Connected Layer"""

    def __init__(self, _input_data, _neurons, _is_layer_output=False):
        """
        Initialize the Pooling Layer

        Args:
            _input_data: input data for the fully connected layer
            _neurons: Layer neurons
        """

        # Check input correctness
        if _neurons <= 0:
            raise RuntimeError("Layer shall have at least 1 neuron")
        if  len(_input_data.shape) != 4 and len(_input_data.shape) != 2:
            raise RuntimeError("Input shall be the output of a Pooling or FC Layer")

        # Reshape Input -> # Batch/FlattenedVolume
        flat_dimension = 0
        if len(_input_data.shape) == 4:
            flat_dimension = (_input_data.shape[1] * _input_data.shape[2] * _input_data.shape[3]).value
        else:
            flat_dimension = _input_data.shape[1].value
        self.x = tf.reshape(_input_data, [-1, flat_dimension])

        # Weights
        self.W = tf.Variable(tf.truncated_normal([flat_dimension, _neurons], stddev=1) * tf.sqrt(2.0 / flat_dimension), tf.float32)
        self.b = tf.Variable(tf.truncated_normal([_neurons], stddev=0.1), tf.float32)

        # Pooling operation
        if _is_layer_output:
            self.computation = tf.nn.relu(tf.add(tf.matmul(self.x, self.W), self.b))
        else:
            self.computation = tf.add(tf.matmul(self.x, self.W), self.b)



class ConvNetwork(trainer.Trainer):
    """Trainable Convolutional Network"""

    def __init__(self, _structure):
        """
        Initialize the Convolutional Network

        Args:
            _structure: Shape of the network layers
                        Each value in the List shall be one of the following
                        objects:
                            - ConvLayer
                            - PoolingLayer

        Raise:  RuntimeError if the passed structure is not compliant.
        """

        # Runtime error if the network structure is not compliant
        if not isinstance(_structure, list):
            raise RuntimeError("Bad Convolutional Network structure")

        def CheckLayerValidity(_layer):
            """Check if a layer is of one of the accepted types"""
            return isinstance(_layer, (ConvLayer, PoolingLayer, FCLayer))

        # Create the Generic Trainer
        super().__init__()

        # Create the network layers
        self.network_layers = []
        for layer in _structure:
            if CheckLayerValidity(layer):
                self.network_layers.append(layer)
            else:
                raise RuntimeError("Bad Layer on structure")

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
                                       feed_dict={self.network_layers[0].input: data, self.truth: label}))

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

            predictions = self.tf_session.run([self.prediction], feed_dict={self.network_layers[0].input: data})
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

        W = np.zeros([_weights.shape[3], _weights.shape[0] * _weights.shape[1]])
        for img_row in range(_weights.shape[0]):
            for img_col in range(_weights.shape[1]):
                for img_channel in range(_weights.shape[2]):
                    for img_pixel in range(_weights.shape[3]):
                        W[img_pixel][img_row * _weights.shape[1] + img_col] += _weights[img_row][img_col][img_channel][img_pixel]

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


