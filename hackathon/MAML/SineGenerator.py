from matplotlib import pyplot as plt
import numpy as np


class SinusoidFunction():
    """
        Wrapper class to generate sinusoid datasets, where the amplitude a varies within [0.1, 5.0]
        and the phase b varies within [0, pi], i.e. the sines are of the form y(x) = a * sin(x + b)
    """

    def __init__(self, K=10, amplitude=None, phase=None):
        """
        Constructor.
        :param K: Number of samples per data set, i.e. K-shot learning task
        :param amplitude:
        :param phase:
        """
        self.K = K
        self.amplitude = amplitude if amplitude else np.random.uniform(0.1, 5.0)
        self.phase = phase if amplitude else np.random.uniform(0, np.pi)
        self.sampled_points = None
        self.x = self._sample_x()

    def _sample_x(self):
        return np.random.uniform(-5, 5, self.K)

    def f(self, x):
        '''Sinewave function.'''
        return self.amplitude * np.sin(x - self.phase)

    def batch(self, x=None, force_new=False):
        '''Returns a batch of size K.

        It also changes the sape of `x` to add a batch dimension to it.

        Args:
            x: Batch data, if given `y` is generated based on this data.
                Usually it is None. If None `self.x` is used.
            force_new: Instead of using `x` argument the batch data is
                uniformly sampled.

        '''
        if x is None:
            if force_new:
                x = self._sample_x()
            else:
                x = self.x
        y = self.f(x)
        return x[:, None], y[:, None]

    def equally_spaced_samples(self, K=None):
        '''Returns `K` equally spaced samples.'''
        if K is None:
            K = self.K
        return self.batch(x=np.linspace(-5, 5, K))


def plot_sines(data, *args, **kwargs):
    '''Plot helper.'''
    x, y = data
    return plt.plot(x, y, *args, **kwargs)


def generate_dataset(K, train_size=20000, test_size=10):
    '''Generate train and test dataset.

    A dataset is composed of SinusoidGenerators that are able to provide
    a batch (`K`) elements at a time.
    '''

    def _generate_dataset(size):
        return [SinusoidFunction(K=K) for _ in range(size)]

    return _generate_dataset(train_size), _generate_dataset(test_size)