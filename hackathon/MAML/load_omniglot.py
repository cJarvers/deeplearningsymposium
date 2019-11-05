import numpy as np
from sklearn.preprocessing import OneHotEncoder
np.random.seed(2191)


class OmniglotNShotDataset():
    def __init__(self, classes_per_set=10, samples_per_class=10, trainsize=1000, valsize=100):

        """
        Constructs an N-Shot omniglot Dataset
        :param batch_size: Experiment batch_size
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
        """
        if samples_per_class > 20:
            print("WARNING: There are only 20 samples per class, but you requested %d." %samples_per_class,
                  "Retrieving 20 samples per class.")
            samples_per_class = 20
        # self.x = np.load("../data/omniglot/data.npy")
        self.x = np.load("data.npy")
        self.x = np.reshape(self.x, [-1, 20, 28, 28, 1])
        print("xshape", self.x.shape)
        shuffle_classes = np.arange(self.x.shape[0])
        np.random.shuffle(shuffle_classes)
        self.x = self.x[shuffle_classes]
        self.x_train, self.x_val = self.x[:1200], self.x[1200:]
        self.normalize()

        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class

        self.indexes = {"train": 0, "val": 0}
        self.datasets = {"train": self.x_train, "val": self.x_val}
        self.datasets_cache = {"train": self.packslice(self.datasets["train"], trainsize),
                               "val": self.packslice(self.datasets["val"], valsize)}

    def normalize(self):
        """
        Normalizes data to have a mean of 0 and stddev of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        print("train_shape", self.x_train.shape, "val_shape", self.x_val.shape)
        print("before_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_val = (self.x_val - self.mean) / self.std
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        print("after_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def packslice(self, data_pack, numsamples):
        """
        Collects numsamples batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        n_samples = self.samples_per_class * self.classes_per_set
        support_cacheX = []
        support_cacheY = []
        print("Got Data Pack wit shape", data_pack.shape)
        for _ in range(numsamples):
            slice_x = np.zeros((n_samples, 28, 28, 1))
            slice_y = np.zeros((n_samples,))

            ind = 0
            pinds = np.random.permutation(n_samples)
            classes = np.random.choice(data_pack.shape[0], self.classes_per_set, False)  # chosen classes

            x_hat_class = np.random.randint(self.classes_per_set)  # target class

            for j, cur_class in enumerate(classes):  # each class
                example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class, False)

                for eind in example_inds:
                    slice_x[pinds[ind], :, :, :] = data_pack[cur_class][eind]
                    slice_y[pinds[ind]] = j
                    ind += 1

                # if j == x_hat_class:
                #     slice_x[n_samples, :, :, :] = data_pack[cur_class][np.random.choice(data_pack.shape[1])]
            support_cacheX.append(slice_x)
            oh = OneHotEncoder(n_values=self.classes_per_set, sparse=False)
            y_oh = oh.fit_transform(np.reshape(slice_y, newshape=(-1, 1)))
            support_cacheY.append(y_oh)

        return np.array(support_cacheX), np.array(support_cacheY)