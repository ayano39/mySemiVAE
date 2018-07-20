import os
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
        # Return numpy arrays
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

    def load_original_mnist(self, f_name, valid_size=1000):
        train, valid, test = self.read_pickle_file(f_name, if_semi=False)
        train_data = self.load_data_with_label(*train)
        valid_data = self.load_data_with_label(*valid[:valid_size])
        test_data = self.load_data_with_label(*test)
        self.valid_size = valid_size
        return train_data, valid_data, test_data

    def load_semi_mnist(self, f_name, valid_size=1000):
        train, valid, test = self.read_pickle_file(f_name, if_semi=True)
        if not self.pure_unsupervised:
            l_train_data = self.load_data_with_label(train[0], train[1])
        else:
            l_train_data = self.load_data_without_label(train[0])
        u_train_data = self.load_data_without_label(train[2])
        valid_data = self.load_data_with_label(*valid[:valid_size])
        test_data = self.load_data_with_label(*test)
        return l_train_data, u_train_data, valid_data, test_data
