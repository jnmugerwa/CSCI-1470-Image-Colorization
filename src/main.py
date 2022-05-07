import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage import color

from model import ColorizationModel
from preprocess import preprocess


def train(model, X, Y):
    """
    Runs through one epoch - all training examples.
    :param model: The initialized Colorization model
    :param X: An np-array of black-and-white images with shape (num_images, 32, 32, 1)
    :param Y: An np-array of color images with shape (num_images, 32, 32, 3)
    :return: None
    """
    # TODO

    num_inputs = X.shape[0]
    num_batches = num_inputs // (model.batch_size)
    indices = list(range(num_inputs))
    indices = tf.random.shuffle(indices)
    tf.gather(X, indices)
    tf.gather(Y, indices)
    inputs = tf.image.random_flip_left_right(X)

    # training model on batches 
    for i in range(num_batches):
        startindex = int(i * model.batch_size)
        endindex = int((i + 1) * model.batch_size)
        input_batch = inputs[startindex:endindex]
        labels_batch = Y[startindex:endindex]

        # make sure to use tf.stack to stack the grayscale and ab channels 
        with tf.GradientTape() as tape:
            # TODO: Remove or keep, if my changes are wrong
            # probs = model.call(input_batch)
            # probs = tf.stack(input_batch, probs)

            # I believe that our problem can be posed as predicting the (a, b) channels given the L channel. So, we
            # no longer need probabilities since we're doing regression.
            predicted_ab_channels = model.call(input_batch)
            true_ab_channels = labels_batch[:, :, :, 1:]
            loss = model.loss_function(predicted_ab_channels, true_ab_channels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, X, Y):
    """
    Runs through one epoch - all testing examples.
    :param model: The initialized Colorization model
    :param X: An np-array of black-and-white images with shape (num_images, 32, 32, 1)
    :param Y: An np-array of color images with shape (num_images, 32, 32, 3)
    :return: model accuracy
    """
    # TODO
    return None


def visualize_results(input, prediction):
    """
    Visualization of model prediction (e.g. input and prediction).
    :param input: Image with only one channel (L); a black and white image. Dims of (32, 32, 1)
    :param prediction: Image with (a, b) channels; the color portion of the image. Dims of (32, 32, 2)
    :return: None
    """
    full_channel_img = np.concatenate((prediction, input), axis=2)
    as_rgb = color.lab2rgb(full_channel_img)
    plt.imshow(as_rgb)
    plt.show()


def main():
    print("Running preprocessing...")
    X_train, X_test, Y_train, Y_test = preprocess()
    print("Preprocessing complete.")

    model = ColorizationModel()

    # TODO
    # Train and Test Model for 20 epochs; 20 is arbitrary, change if necessary
    num_epochs = 20
    for _ in range(num_epochs):
        train(model, X_train, Y_train)

    acc = test(model, X_test, Y_test)
    print(f"Accuracy of Colorization model: {acc}")


if __name__ == '__main__':
    main()
