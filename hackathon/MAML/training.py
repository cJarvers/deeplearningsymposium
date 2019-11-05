import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
from helper import *
from RegressionModel import *
import random

def loss_function(pred_y, y):
    """
    Mean Squared Error Loss Function
    :param pred_y: predicted y value
    :param y: true y value
    :return: MSE between y_pred and y
    """
    return keras_backend.mean(keras.losses.mean_squared_error(y, pred_y))


def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)


def compute_loss(model, x, y, loss_fn=loss_function):
    """
    Computes the total loss of the model and one forward pass
    :param model:
    :param x: inputs
    :param y: outputs
    :param loss_fn: loss function to use
    :return: tuple of total loss and model output
    """
    y_pred = model.forward(x)
    mse = loss_fn(y, y_pred)
    return mse, y_pred


def compute_gradients(model, x, y, loss_fn=loss_function):
    """
    Compute the gradients for a given model wrt to a given loss function
    :param model: model to compute grads for
    :param x: input
    :param y: output
    :param loss_fn: loss function
    :return: tuple of gradient tape and loss
    """
    with tf.GradientTape() as tape:
        loss, _ = compute_loss(model, x, y, loss_fn)
    return tape.gradient(loss, model.trainable_variables), loss


def apply_gradients(optimizer, gradients, variables):
    """
    Applies the gradient to network parameters
    :param optimizer: optimizer to use, e.g. adam
    :param gradients: gradient tape
    :param variables: variables to update
    :return: none
    """
    optimizer.apply_gradients(zip(gradients, variables))


def train_batch(x, y, model, optimizer):
    """
    Train model on a batch of data
    :param x: input data
    :param y: outputs
    :param model: model to use
    :param optimizer: opimizer to use
    :return: loss of the training step
    """
    tensor_x, tensor_y = np_to_tensor((x, y))
    gradients, loss = compute_gradients(model, tensor_x, tensor_y)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    return loss


def train_model(dataset, epochs=1, lr=0.001, log_steps=1000):
    """
    Train model for a certain amount of epochs
    :param dataset: dataset for training
    :param epochs: epochs (runs through the dataset)
    :param lr: learning rate
    :param log_steps: logging interval
    :return: trained model
    """
    model = RegressionModel()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        losses = []
        total_loss = 0
        for i, sinusoid_generator in enumerate(dataset):
            x, y = sinusoid_generator.batch()
            loss = train_batch(x, y, model, optimizer)
            total_loss += loss
            curr_loss = total_loss / (i + 1.0)
            losses.append(curr_loss)

            if i % log_steps == 0 and i > 0:
                print('Step {}: loss = {}, Time to run {} steps = {:.2f} seconds'.format(i, curr_loss))
        plt.plot(losses)
        plt.title('Loss Vs Time steps')
        plt.show()
    return model


def train(model, epochs, dataset, lr_inner=0.01, batch_size=1, log_steps=1000):
    """
    Train a model using the MAML algorithm
    :param model: model to train
    :param epochs: number of epochs (i.e. passed through the whole dataset)
    :param dataset: dataset to train on
    :param lr_inner: learning rate of inner updates
    :param batch_size: batch size for updates
    :param log_steps: logging interval
    :return:
    """
    optimizer = keras.optimizers.Adam()

    # Step 2: instead of checking for convergence, we train for a number
    # of epochs
    print("Traning MAML for %d Epochs" % epochs)
    for ei in range(epochs):
        print("Training Progress: %d/%d" % (ei, epochs))
        total_loss = 0
        losses = []
        # Step 3 and 4
        for i, t in enumerate(random.sample(dataset, len(dataset))):
            x, y = np_to_tensor(t.batch())
            model.forward(x)  # run forward pass to initialize weights
            with tf.GradientTape() as test_tape:
                # test_tape.watch(model.trainable_variables)
                # Step 5
                with tf.GradientTape() as train_tape:
                    train_loss, _ = compute_loss(model, x, y)
                # Step 6
                gradients = train_tape.gradient(train_loss, model.trainable_variables)
                k = 0
                model_copy = copy_model(model, x)
                for j in range(len(model_copy.layers)):
                    model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                                                              tf.multiply(lr_inner, gradients[k]))
                    model_copy.layers[j].bias = tf.subtract(model.layers[j].bias,
                                                            tf.multiply(lr_inner, gradients[k + 1]))
                    k += 2
                # Step 8
                test_loss, logits = compute_loss(model_copy, x, y)
            # Step 8
            gradients = test_tape.gradient(test_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Logs
            total_loss += test_loss
            loss = total_loss / (i + 1.0)
            losses.append(loss)

            if i % log_steps == 0 and i > 0:
                print('Step {}: loss = {}'.format(i, loss))
        plt.plot(losses)
        plt.show()


def copy_model(model, x):
    """
    Creates a copy of a model
    :param model: model to copy
    :param x: input
    :return: copy of the model
    """
    copied_model = RegressionModel()
    # One forward pass is required to initialize parameters
    copied_model.forward(tf.convert_to_tensor(x))
    copied_model.set_weights(model.get_weights())
    return copied_model


def eval_sine_test(model, optimizer, x, y, x_test, y_test, num_steps=(0, 1, 10)):
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


def eval_sinewave_for_test(model, sinusoid_generator=None, num_steps=(0, 1, 10), lr=0.01, plot=True):
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
    fit_res = eval_sine_test(copied_model, optimizer, x, y, x_test, y_test, num_steps)

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