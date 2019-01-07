import numpy as np
import joblib


class Dataset(object):
    def __init__(self, filepath, statebatchsize, mode=None, imsize=None):
        images, s1, s2, labels = joblib.load(filepath)

        cut = int(len(images) * 1 / 7)
        if mode == 'train':
            part = np.s_[cut:]
        elif mode == 'test':
            part = np.s_[:cut]
        else:
            raise ValueError('mode can be either train or test.')

        self._images = np.array(images).astype(float)[part]
        self._s1 = np.array(s1).astype(float)[part]
        self._s2 = np.array(s2).astype(float)[part]

        self._labels = np.array(labels).astype(float)[part]  # .flatten()
        self._num_examples = self._images.shape[0]

        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._statebatchsize = statebatchsize
        print(np.shape(self._labels))
        print(np.shape(self._images))

    #@property
    def images(self):
        return self._images

    #@property
    def s1(self):
        return self._s1

    #@property
    def s2(self):
        return self._s2

    #@property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return next 'batch_size' examples from this data set.
        """
        # Check: batch size should not exceed the size of dataset
        assert batch_size <= self._num_examples

        # Initial index for slicing
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        # Not enough data for a batch: Reset + Shuffling
        if self._index_in_epoch > self._num_examples:
            # Increment finished epoch
            self._epochs_completed += 1
            # Shuffule the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._s1 = self._s1[perm]
            self._s2 = self._s2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        # End index for slicing
        end = self._index_in_epoch
        #print(start, end,np.shape(self._labels[start * self._statebatchsize:end * self._statebatchsize]))
        return self._images[start:end], self._s1[start:end], self._s2[start:end], self._labels[start:end].flatten()
