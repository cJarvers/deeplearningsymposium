from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from training import *
import numpy as np


def main():
    # Reproduction
    np.random.seed(1234)
    for _ in range(3):
        plt.title('Sinusoid examples')
        plot_sines(SinusoidFunction(K=100).equally_spaced_samples())
    plt.show()
    train_ds, test_ds = generate_dataset(K=10, train_size=1000)
    model = RegressionModel()
    train(model, 1, train_ds)
    eval_sinewave_for_test(model=model)


if __name__ == "__main__":
    main()

