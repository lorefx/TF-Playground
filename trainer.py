"""
This module will provide a base class for a trainer.
A trainer provides all facilites to configure, run and collect
performances for a network
"""

from matplotlib import pyplot as plt
import tensorflow as tf


class Trainer(object):
    """
    A trainer provides all facilites to configure, run and collect
    performances for a network
    """

    def __init__(self):
        """Create a Trainer Manager"""

        # Network Configuration variables
        self.global_step = None
        self.learning_rate = None
        self.truth = None
        self.cross_entropy = None
        self.train = None
        self.val_predictions = None
        self.val_labels = None
        self.accuracy = None

        # Plottable Data
        self.plottable_data = {'epoch': [], 'loss': [], 'train': [], 'test': [], 'learnrate': []}

        # Epochs Performance Management
        self.cur_epoch = 0
        self.best_accuracy_epoch = 0
        self.last_try_epochs = 0
        self.min_epochs_without_progress = 0
        self.best_accuracy = 0
        self.best_weights = []


    def ConfigureTraining(self, _learning_rate, _decay_steps, _decay_amount, _min_epochs_without_progress=0):
        """
        Configure the Neural Network training

        Args:
            _learning_rate:  Staring Learning Rate
            _decay_steps: amount of steps to apply a decay
            _decay_amount: percentage of the previous learning rate to mantain ageter decay steps
            _min_epochs_without_progress: Minimum number of epochs without progress to run
        """
        raise NotImplementedError("Subclass must implement abstract method ConfigureTraining")


    def StartTraining(self):
        """
        Starts the training operations
        """
        # Initialize the Tensorflow Session
        self.init = tf.global_variables_initializer()
        self.tf_session = tf.Session()

        self.tf_session.run(self.init)


    def StopTraining(self):
        """
        Starts the training operations
        """
        self.tf_session.close()


    def RunTrainingStep(self, _input_data):
        """
        Exeute a training step of the network

        Args:
            _input_data: input data for the input layer

        Returns: loss value
        """
        raise NotImplementedError("Subclass must implement abstract method RunTrainingStep")


    def RunPrediction(self, _input_data, _batch_size):
        """
        Exeute a prediction step of the network

        Args:
            _input_data: input data for the input layer
            _batch_size: number of data to process in each iteration

        Returns: prediction accuracy
        """
        raise NotImplementedError("Subclass must implement abstract method RunPrediction")


    def CheckTrainingEnd(self):
        """
        Chech if the training has ended or not

        Returns:
            True if the Training has ended, False otherwise
        """
        return self.cur_epoch > (self.best_accuracy_epoch + self.last_try_epochs + self.min_epochs_without_progress)


    def EpochStart(self):

        """Starts the current Epoch"""
        self.images = 0
        self.train_batches = 0
        self.train_loss = 0
        self.train_accuracy = 0


    def EpochEnd(self):
        """
        End the current Epoch
        """

        # Retrieve the current learning rate
        actual_learning_rate = self.tf_session.run(self.learning_rate)

        # Update Plottable Data
        self.plottable_data['loss'].append(self.train_loss / self.images)
        self.plottable_data['learnrate'].append(actual_learning_rate)

        self.plottable_data['epoch'].append(self.cur_epoch)
        self.cur_epoch += 1


    def AddTrainingResult(self, _loss, _batch_size):
        """
        Add the results for a single batch

        Args:
            _loss: Batch loss
            _batch_size: Batch size
        """
        self.train_loss += _loss
        self.images += _batch_size


    def RecordAccuracy(self, _train_accuracy, _test_accuracy):
        """
        Add the results for a test run

        Args:
            _train_accuracy: train accuracy
            _test_accuracy: test accuracy

        Returns:
            True if the results are the best ones, False otherwise
        """
        # Check if the test accuracy is the best one and update the
        # proper markers
        print("Epoch ", self.cur_epoch, " | Accuracy ", _test_accuracy * 100, " | Training ", _train_accuracy * 100)
        is_best_epoch = False
        if _test_accuracy > self.best_accuracy:
            self.best_accuracy = _test_accuracy
            self.last_try_epochs = self.cur_epoch -self. best_accuracy_epoch
            self.best_accuracy_epoch = self.cur_epoch
            is_best_epoch = True

        self.plottable_data['test'].append(_test_accuracy * 100)
        self.plottable_data['train'].append(_train_accuracy * 100)

        return is_best_epoch


    def DisplayResults(self):
        """Outputs the training results"""

        # Best result output
        print("Best Accuracy ", self.best_accuracy * 100, " @ Epoch ", self.best_accuracy_epoch)

        # Results plotting
        plt.subplot(311)
        plt.plot(self.plottable_data['epoch'], self.plottable_data['test'], 'b', self.plottable_data['epoch'], self.plottable_data['train'], 'g')
        plt.subplot(312)
        plt.plot(self.plottable_data['epoch'], self.plottable_data['loss'], 'r')
        plt.subplot(313)
        plt.plot(self.plottable_data['epoch'], self.plottable_data['learnrate'], 'b')
        plt.show()


    def ProcessWeightsForDisplay(self, _weights, _width, _height):
        """
        Setup the weights to be displayed

        Args:
            _weights: weights to display
            _width: display image width
            _height: display image height
        """
        raise NotImplementedError("Subclass must implement abstract method ProcessWeightsForDisplay")


    def DisplayPlot(self, _weights, _plot_rows, _plot_cols):
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

        # Plot each slice as an independent subplot
        fig, axes = plt.subplots(nrows=_plot_rows, ncols=_plot_cols)
        for dat, ax in zip(_weights, axes.flat):
            # The vmin and vmax arguments specify the color limits
            im = ax.imshow(dat, vmin=_weights.min(), vmax=_weights.max(), cmap="hot")

        # Make an axis for the colorbar on the right side
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        fig.colorbar(im, cax=cax)
        plt.show()
        