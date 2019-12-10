import numpy as np
import tensorflow as tf
from io import StringIO
from tensorflow.python.lib.io import file_io
from architecture import Architecture
from emnist import extract_training_samples

class DatasetLoader:

    def load_dataset(self, config):
        images, labels = self._load_data(config)

        batch_size = config.batch_size
        num_batches = int(np.ceil(images.shape[0] / float(batch_size)))

        X = self._data_tensor(images)
        dataset = tf.data.Dataset.from_tensor_slices((X, labels))
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)
        return dataset, num_batches

    ### Private ###

    def _load_data(self, config):

        # open data file
        # f = StringIO(file_io.read_file_to_string(config.data_dir))
        # mnist = np.load(f)  SAMPLE_DIR
        

        # # Return numpy arrays of shapes (M, 28, 28), (M,)
        # x_train, y_train = mnist['x_train'], mnist['y_train']
        x_train, y_train = extract_training_samples('letters')
        return x_train, y_train


    def _data_tensor(self, images):
        '''
        images: (M, 28, 28), values in range [0, 255]
        returns: tensor of images shaped [M, 32, 32, 1], with values in range [-1, 1]
        '''

        # Turn numpy array into a tensor X of shape [M, 28, 28, 1].
        X = tf.constant(images)
        X = tf.reshape(X, [-1, 28, 28, 1])

        # resize images to img_size
        img_size = Architecture.img_size
        if img_size == 28:
            X = tf.image.resize_images(X, [29, 29]) # an awkward way of forcing some type converting
        X = tf.image.resize_images(X, [img_size, img_size])

        # The data is currently in a range [0, 255].
        # Transform data to have a range [-1, 1].
        # We do this to match the range of tanh, the activation on the generator's output layer.
        X = X / 128.
        X = X - 1.
        return X



