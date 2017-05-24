"""
This module will provide a Loader for the MNIST dataset
provided by the LeCun website at http://yann.lecun.com/exdb/mnist/
"""

import loader
import numpy as np


class MnistImage(object):
    """MNIST Image Descriptor"""

    def __init__(self, _width, _height, _data=None):
        """
        Create an MNIST Image Descriptor

        Args:
            _width: Image Width
            _height: Image Height
            _data: Image pixel data
        """
        self.width = _width
        self.height = _height

        if _data is None:
            #_data = np.empty(_width * _height, dtype=int)
            _data = []
        self.data = _data



class MnistLabel(object):
    """MNIST Label Descriptor"""

    def __init__(self, _class):
        """
        Create an MNIST Label Descriptor

        Args:
            _class: Label Class (classification category)
        """
        # Convert to One Hot Encoding
        self.label = np.zeros(10, dtype=int)
        self.label[int.from_bytes(_class, byteorder='little')] = 1



class LoaderMnist(loader.Loader):
    """
    MNIST Loader from LeCun format

    Loader that creates train, validation and test datasets from the LeCun MNIST datasets
    http://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, _train_data_file, _train_label_file, _test_data_file, _test_label_file, _validation_percent=0, _limit_images=0):
        """
        Load and parse the provided mnist files to create the datasets

        Args:
            _train_data_file: path of the train dataset file
            _train_label_file: path of the train label file
            _test_data_file: path of the test dataset file
            _test_label_file: path of the test label file
            _validation_percent: percent of the training set to be converted as validation. Range [0,99]
            _limit_images: maximum number of images to load

        Raises:
            RuntimeError: An error occurred creating the Loader object
        """
        super().__init__()
        try:

            # Retrieve the train data
            train_data = self.ParseDataFile(_train_data_file, _limit_images)
            train_label = self.ParseLabelFile(_train_label_file, _limit_images)

            if train_data['num_data'] != train_label['num_data']:
                raise RuntimeError("Training data and labels number are different")

            # Retrieve the test data
            test_data = self.ParseDataFile(_test_data_file, _limit_images)
            test_label = self.ParseLabelFile(_test_label_file, _limit_images)

            if test_data['num_data'] != test_label['num_data']:
                raise RuntimeError("Test data and labels number are different")

            # Prepare the Training Set
            if _validation_percent < 0:
                _validation_percent = 0
            elif _validation_percent > 99:
                _validation_percent = 99

            validation_elements = int(train_data['num_data'] / 100 * _validation_percent)
            self.train_set = []
            self.validation_set = []

            print("Preparing Training Set")
            for index in range(train_data['num_data'] - validation_elements):
                self.train_set.append({'data': train_data['data'][index].data, 
                                       'label': train_label['data'][index].label})

            if (validation_elements > 0):
                print("Preparing Validation Set")
                for index in range(train_data['num_data'] - validation_elements, train_data['num_data']):
                    self.validation_set.append({'data': train_data['data'][index].data, 
                                                'label': train_label['data'][index].label})
            # Prepare the Test Set
            self.test_set = []
            print("Preparing Test Set")
            for index in range(test_data['num_data']):
                self.test_set.append({'data': test_data['data'][index].data,
                                      'label': test_label['data'][index].label})

            print("Training Set size: ", len(self.train_set))
            print("Validation Set size: ", len(self.validation_set))
            print("Test Set size: ", len(self.test_set))

            # Prepare Instance Data
            self.batch_available_num = len(self.train_set)

        except Exception as exception:
            print(exception)
            raise RuntimeError("Unable to create the LoaderMnist object")


    def ParseU32(self, _file):
        """
        Retrieve an uint32 value from file

        Args:
            _file: already opened file to parse
        """
        data = _file.read(4)
        return data[0] * pow(16, 6) + data[1] * pow(16, 4) + data[2] * pow(16, 2) + data[3]


    def ParseU8(self, _file):
        """
        Retrieve an uint8 value from file

        Args:
            _file: already opened file to parse
        """
        return _file.read(1)


    def GetBatch(self, _data_num):
        """
        Retrieve a batch of training data

        Args:
            _data_num: number of data requested

        Returns:
            - batch completed: bool value
            - The requested number of data if enough are available,
              otherwise the maximum number available
        """

        # Prepare the batch
        retrieve_num = min(_data_num, self.batch_available_num)
        start = len(self.train_set) - self.batch_available_num
        stop = start + retrieve_num
        batch = self.train_set[start : stop]
        self.batch_available_num -= retrieve_num

        # Check if a randomization of the batch is necessary
        batch_completed = False
        if self.batch_available_num == 0:
            self.train_set= np.random.permutation(self.train_set)
            self.batch_available_num = len(self.train_set)
            batch_completed = True

        return batch_completed, batch


    def ParseDataFile(self, _data_file, _limit_images):
        """
        Parse and retrieve data from a dataset file

        Args:
            _data_file: path of file to parse
            _limit_images: maximum number of images to load

        Returns:
            The parsed data object structured as a dictionary with:
                num_data: number of elements returned
                data: MnistImage objects
        """
        res = {'num_data': 0, 'data': None}
        try:
            with open(_data_file, 'rb') as file:

                # Check magic number for Training File
                magic_number = self.ParseU32(file)
                if magic_number != 0x00000803:
                    raise RuntimeError("Magic Number "
                                       + magic_number
                                       + " is not correct for a training data")

                # Parse header data
                image_number = self.ParseU32(file)
                image_height = self.ParseU32(file)
                image_width = self.ParseU32(file)

                if _limit_images > 0:
                    image_number = min(_limit_images, image_number)

                # Parse Data
                pixel_number = image_width * image_height
                data = []
                mean = 0
                for elapsed_images in range(image_number):
                    image = MnistImage(image_width, image_height)
                    data.append(image)
                    for pixel in range(pixel_number):
                        image.data.append(int.from_bytes(self.ParseU8(file), byteorder='little') / 255)
                    mean +=  np.mean(image.data)
                    if (elapsed_images % 1000) == 0:
                        print("Elapsed Images: ", elapsed_images, "/", image_number)

                # Mean center data
                mean /= image_number
                for image in data:
                    image.data -= mean

                res['num_data'] = image_number
                res['data'] = data

        except Exception as exception:
            raise exception

        else:
            return res


    def ParseLabelFile(self, _label_file, _limit_images):
        """
        Parse and retrieve data from a dataset file

        Args:
            _label_file: path of file to parse
            _limit_images: maximum number of images to load

        Returns:
            The parsed data object structured as a dictionary with:
                num_data: number of labels returned
                data: MnistLabel objects

        Raises:
            OSError: An error occurred while handling the file
            RuntimeError: An error occurred while parsing the file
        """
        res = {'num_data': 0, 'data': None}
        try:
            with open(_label_file, 'rb') as file:

                # Check magic number for Training File
                magic_number = self.ParseU32(file)
                if magic_number != 0x00000801:
                    raise RuntimeError("Magic Number "
                                       + magic_number
                                       + " is not correct for a label data")

                # Parse header data
                label_number = self.ParseU32(file)

                if _limit_images > 0:
                    label_number = min(_limit_images, label_number)

                # Parse Data
                data = []
                for elapsed_labels in range(label_number):
                    data.append(MnistLabel(self.ParseU8(file)))

                    if (elapsed_labels % 1000) == 0:
                        print("Elapsed Labels: ", elapsed_labels, "/", label_number)

                res['num_data'] = label_number
                res['data'] = data

        except Exception as exception:
            raise exception

        else:
            return res
