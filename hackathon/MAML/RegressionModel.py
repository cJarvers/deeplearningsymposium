import tensorflow.keras as keras


class RegressionModel(keras.Model):
    """
    Creates a feed forwar neural network for regression. Architecture is as described in the original
    MAML paper.
    """
    def __init__(self):
        super().__init__()
        self.hidden1 = keras.layers.Dense(40, input_shape=(1,), dtype="float32")
        self.hidden2 = keras.layers.Dense(40, dtype="float32")
        self.out = keras.layers.Dense(1, dtype="float32")

    def forward(self, x):
        x = keras.activations.relu(self.hidden1(x))
        x = keras.activations.relu(self.hidden2(x))
        x = self.out(x)
        return x