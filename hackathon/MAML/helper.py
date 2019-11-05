import tensorflow.keras as keras
import tensorflow as tf
from training import *
from SineGenerator import *
from RegressionModel import RegressionModel


def copy_model(model, x):
    # Copy model weights to a new model.

    copied_model = RegressionModel()

    # If we don't run this step the weights are not "initialized"
    # and the gradients will not be computed.
    copied_model.forward(tf.convert_to_tensor(x))

    copied_model.set_weights(model.get_weights())
    return copied_model


def eval_sine(model, optimizer, x, y, x_test, y_test, num_steps=(0, 1, 10)):
    '''Evaluate how the model fits to the curve training for `fits` steps.

    Args:
        model: Model evaluated.
        optimizer: Optimizer to be for training.
        x: Data used for training.
        y: Targets used for training.
        x_test: Data used for evaluation.
        y_test: Targets used for evaluation.
        num_steps: Number of steps to log.
    '''
    fit_res = []

    tensor_x_test, tensor_y_test = np_to_tensor((x_test, y_test))

    # If 0 in fits we log the loss before any training
    if 0 in num_steps:
        loss, logits = compute_loss(model, tensor_x_test, tensor_y_test)
        fit_res.append((0, logits, loss))

    for step in range(1, np.max(num_steps) + 1):
        train_batch(x, y, model, optimizer)
        loss, logits = compute_loss(model, tensor_x_test, tensor_y_test)
        if step in num_steps:
            fit_res.append(
                (
                    step,
                    logits,
                    loss
                )
            )
    return fit_res


def eval_model(model, sinusoid_generator=None, num_steps=(0, 1, 10), lr=0.01, plot=True):
    '''Evaluates how the sinewave addapts at dataset.

    The idea is to use the pretrained model as a weight initializer and
    try to fit the model on this new dataset.

    Args:
        model: Already trained model.
        sinusoid_generator: A sinusoidGenerator instance.
        num_steps: Number of training steps to be logged.
        lr: Learning rate used for training on the test data.
        plot: If plot is True than it plots how the curves are fitted along
            `num_steps`.

    Returns:
        The fit results. A list containing the loss, logits and step. For
        every step at `num_steps`.
    '''

    if sinusoid_generator is None:
        sinusoid_generator = SinusoidFunction(K=10)

    # generate equally spaced samples for ploting
    x_test, y_test = sinusoid_generator.equally_spaced_samples(100)

    # batch used for training
    x, y = sinusoid_generator.batch()

    # copy model so we can use the same model multiple times
    copied_model = copy_model(model, x)

    # use SGD for this part of training as described in the paper
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    # run training and log fit results
    fit_res = eval_sine(copied_model, optimizer, x, y, x_test, y_test, num_steps)

    # plot
    train, = plt.plot(x, y, '^')
    ground_truth, = plt.plot(x_test, y_test)
    plots = [train, ground_truth]
    legend = ['Training Points', 'True Function']
    for n, res, loss in fit_res:
        cur, = plt.plot(x_test, res[:, 0], '--')
        plots.append(cur)
        legend.append('After %d Steps' % n)
    plt.legend(plots, legend)
    plt.ylim(-5, 5)
    plt.xlim(-6, 6)
    if plot:
        plt.show()

    return fit_res