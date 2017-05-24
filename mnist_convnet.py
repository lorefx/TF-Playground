"""
Main file for the MNIST Deep Learning
"""

from loader_mnist import LoaderMnist
import convnet as cvnet


def main():
    """Main program execution"""

    # Dataset Loader
    dset_loader = None

    # Create the Dataset Loader if possible
    try:
        # Configure dataset loading and subdivision ==> USER
        validation_percent = 0
        limit_images = 0
        dset_loader = LoaderMnist("./mnist/train-images-idx3-ubyte",
                                  "./mnist/train-labels-idx1-ubyte",
                                  "./mnist/t10k-images-idx3-ubyte",
                                  "./mnist/t10k-labels-idx1-ubyte",
                                  validation_percent,
                                  limit_images)

    except RuntimeError as exception:
        print(exception)
        raise SystemExit

    # Create the Neural Network
    nnetwork = None
    try:
        # Create the network with its structure ==> USER
        layers = []
        layers.append(cvnet.ConvLayer(None, _input_shape=[28, 28, 1], _kernel_shape=[5, 5, 1, 6], _stride_shape=[1, 1], _is_input_layer=True))
        layers.append(cvnet.PoolingLayer(layers[-1].computation, _pool_shape=[2, 2], _stride_shape=[2, 2]))
        layers.append(cvnet.ConvLayer(layers[-1].computation, _input_shape=[14, 14, 6], _kernel_shape=[5, 5, 6, 16], _stride_shape=[1, 1]))
        layers.append(cvnet.PoolingLayer(layers[-1].computation, _pool_shape=[2, 2], _stride_shape=[2, 2]))
        layers.append(cvnet.FCLayer(layers[-1].computation, 120))
        layers.append(cvnet.FCLayer(layers[-1].computation, 84))
        layers.append(cvnet.FCLayer(layers[-1].computation, 10, _is_layer_output=True))

        nnetwork = cvnet.ConvNetwork(layers)

    except RuntimeError as exception:
        print(exception)
        raise SystemExit

    # Configuration of the Network Hyperparameters ==> USER
    train_batch_size = 128
    test_batch_size = 128
    learning_rate = 0.001
    learning_rate_decay_steps = 1000
    learning_rate_decay_amount = 0.98
    min_epochs_without_progress = 10
    nnetwork.ConfigureTraining(learning_rate, learning_rate_decay_steps, learning_rate_decay_amount, min_epochs_without_progress)

    # Start thensorflow session
    nnetwork.StartTraining()

    # Start epochs training
    while nnetwork.CheckTrainingEnd() is False:

        # Start the current Epoch
        nnetwork.EpochStart()

        # Train batches
        batch_completed = False
        while not batch_completed:
            # Execute a training step on the batch
            batch_completed, cur_batch = dset_loader.GetBatch(train_batch_size)
            nnetwork.RunTrainingStep(cur_batch)

        # Prediction of validation set
        if len(dset_loader.validation_set) > 0:
            nnetwork.AddTestResults(dset_loader.train_set, dset_loader.validation_set, test_batch_size)
        else:
            nnetwork.AddTestResults(dset_loader.train_set, dset_loader.test_set, test_batch_size)

        # Advance to next Epoch
        nnetwork.EpochEnd()

    # Close the Training Session
    nnetwork.StopTraining()

    # Display the training results
    nnetwork.DisplayResults()
    nnetwork.DisplayWeights(nnetwork.best_weights[0], _width=5, _height=5, _plot_rows=2, _plot_cols=3)
    nnetwork.DisplayWeights(nnetwork.best_weights[1], _width=5, _height=5, _plot_rows=4, _plot_cols=4)

    return

# Execute the program
if __name__ == '__main__':
    main()


