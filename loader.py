"""
This module will provide a Loader for the MNIST dataset
provided by the LeCun website at http://yann.lecun.com/exdb/mnist/
"""

class Loader(object):
    """Dataset Loader abstract class"""

    def __init__(self):
        """Loader constructor"""
        pass

    def GetBatch(self, _data_num):
        """
        Retrieve a batch of training data

        Args:
            _data_num: number of data requested

        Returns:
            The requested number of data if enough are available,
            otherwise the maximum number available
        """
        raise NotImplementedError("Subclass must implement abstract method GetBatch")
