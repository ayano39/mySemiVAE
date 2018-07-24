import gzip
import pickle
import numpy as np
from collections import defaultdict

fin_name = "dataset/mnist_28.pkl.gz"
fout_name = "dataset/mnist_l{}_u{}.pkl.gz"
fixed_size = 40000
settings = [0, 10, 100, 1000, 5000]


class DataSampler(object):
    def __init__(self, x_all, y_all):
        self.num_types = None
        self.label_to_x_array = None
        self._sort_into_labels(x_all, y_all)

    def _sort_into_labels(self, x_all, y_all):
        label_to_x_list = defaultdict(list)
        for x, y in zip(x_all, y_all):
            label_to_x_list[y].append(x)
        self.num_types = len(label_to_x_list)
        self.label_to_x_array = [np.asarray(label_to_x_list[i])
                                 for i in range(self.num_types)]
        shuffler = lambda array: np.random.shuffle(array)
        map(shuffler, self.label_to_x_array)

    def pack_equally(self, pack_size, with_label, pack_and_remove=False):
        if pack_size == 0:
            if with_label:
                return np.array([], np.float32), np.array([], np.float32)
            else:
                return np.array([], np.float32)

        assert pack_size % self.num_types == 0, "pack_size % num_types != 0"
        size_per_type = pack_size // self.num_types

        x_list = []
        y_list = []
        for i in range(self.num_types):
            x_list.append(self.label_to_x_array[i][:size_per_type])
            y_list.extend([i] * size_per_type)
            if pack_and_remove:
                self.label_to_x_array[i] = self.label_to_x_array[i][size_per_type:]

        if with_label:
            shuffle_indices = np.random.permutation(pack_size)
            x = np.concatenate(x_list, axis=0)[shuffle_indices]
            y = np.asarray(y_list)[shuffle_indices]
            return x, y
        else:
            x = np.concatenate(x_list, axis=0)
            np.random.shuffle(x)
            return x


def read_pickle_file(f_name):
    f = gzip.open(f_name, "rb")
    train, valid, test = pickle.load(f, encoding="bytes")
    return train, valid, test


def sample_fix_labelled(labelled_size, unlabelled_settings):
    train, valid, test = read_pickle_file(fin_name)
    data_sampler = DataSampler(*train)
    x_labelled, y_labelled = data_sampler.pack_equally(labelled_size,
                                                       with_label=True,
                                                       pack_and_remove=True)
    for unlabelled_size in unlabelled_settings:
        f_name = fout_name.format(labelled_size, unlabelled_size)
        x_unlabelled = data_sampler.pack_equally(unlabelled_size, with_label=False)
        with gzip.open(f_name, "wb") as f:
            train_repack = (x_labelled, y_labelled, x_unlabelled)
            pickle.dump((train_repack, valid, test), f)


def sample_fix_unlabelled(unlabelled_size, labelled_settings):
    train, valid, test = read_pickle_file(fin_name)
    data_sampler = DataSampler(*train)
    x_unlabelled = data_sampler.pack_equally(unlabelled_size,
                                             with_label=False,
                                             pack_and_remove=True)

    for labelled_size in labelled_settings:
        f_name = fout_name.format(labelled_size, unlabelled_size)
        x_labelled, y_labelled = data_sampler.pack_equally(labelled_size,
                                                           with_label=True)
        with gzip.open(f_name, "wb") as f:
            train_repack = (x_labelled, y_labelled, x_unlabelled)
            pickle.dump((train_repack, valid, test), f)


if __name__ == "__main__":
    #sample_fix_labelled(fixed_size, settings)
    sample_fix_unlabelled(fixed_size, settings)
