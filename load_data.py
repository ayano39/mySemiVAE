import gzip
import pickle
import tensorflow as tf


class DataLoader(object):
    def __init__(self):
        self.train_size = None
        self.valid_size = None
        self.test_size = None
        self.pure_supervised = False
        self.pure_unsupervised = False

    def read_pickle_file(self, f_name, if_semi=False):
        # When 'if_semi' is False, the data is assumed to be structured like
        # ((train_x, train_y), (valid_x, valid_y), (test_x, test_y); Otherwise,
        # the data will be ((train_labelled_x, train_labelled_y, train_unlabelled_x),
        #  (valid_x, valid_y), (test_x, test_y))
        # Return: numpy arrays
        f = gzip.open(f_name, 'rb')
        train, valid, test = pickle.load(f, encoding="bytes")
        if if_semi:
            self.train_size = train[0].shape[0] + train[2].shape[0]
            if train[0].shape[0] == 0:
                self.pure_unsupervised = True
            if train[2].shape[0] == 0:
                self.pure_supervised = True
        else:
            self.train_size = train[0].shape[0]
        self.valid_size = valid[0].shape[0]
        self.test_size = test[0].shape[0]
        return train, valid, test

    def load_data_with_label(self, image_array, label_array):
        def _make_onehot(label):
            return tf.one_hot(label, 10)

        images = tf.data.Dataset.from_tensor_slices(image_array)
        labels = tf.data.Dataset.from_tensor_slices(label_array)\
            .map(_make_onehot)
        dataset = tf.data.Dataset.zip((images, labels))
        return dataset

    def load_data_without_label(self, image_array):
        dataset = tf.data.Dataset.from_tensor_slices(image_array)
        return dataset

    def load_original_mnist(self, f_name):
        train, valid, test = self.read_pickle_file(f_name, if_semi=False)
        train_data = self.load_data_with_label(*train)
        valid_data = self.load_data_with_label(*valid)
        test_data = self.load_data_with_label(*test)
        return train_data, valid_data, test_data

    def load_semi_mnist(self, f_name):
        train, valid, test = self.read_pickle_file(f_name, if_semi=True)
        if self.pure_unsupervised:
            l_train_data = tf.data.Dataset.from_tensor_slices((
                tf.constant(0, dtype=tf.float32, shape=[]),
                tf.constant(0, dtype=tf.float32, shape=[])
            ))
        else:
            l_train_data = self.load_data_with_label(train[0], train[1])

        if self.pure_supervised:
            u_train_data = None
        else:
            u_train_data = self.load_data_without_label(train[2])

        valid_data = self.load_data_with_label(*valid)
        test_data = self.load_data_with_label(*test)
        return l_train_data, u_train_data, valid_data, test_data
