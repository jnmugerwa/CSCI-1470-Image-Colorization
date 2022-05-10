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
    num_inputs = X.shape[0]
    num_batches = num_inputs // (model.batch_size)
    indices = list(range(num_inputs))
    indices = tf.random.shuffle(indices)
    tf.gather(X, indices)
    tf.gather(Y, indices)
    inputs = tf.image.random_flip_left_right(X)

    for i in range(num_batches):
        startindex = int(i * model.batch_size)
        endindex = int((i + 1) * model.batch_size)
        input_batch = inputs[startindex:endindex]
        labels_batch = Y[startindex:endindex]

        with tf.GradientTape() as tape:
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
    :return: model accuracy; average loss per batch
    """
    num_inputs = X.shape[0]
    num_batches = num_inputs // (model.batch_size)
    losses = []

    for i in range(num_batches):
        startindex = int(i * model.batch_size)
        endindex = int((i + 1) * model.batch_size)
        input_batch = X[startindex:endindex]
        labels_batch = Y[startindex:endindex]

        predicted_ab_channels = model.call(input_batch)
        true_ab_channels = labels_batch[:, :, :, 1:]

        loss = model.loss_function(predicted_ab_channels, true_ab_channels)
        losses.append(loss)

    return tf.reduce_mean(losses)


def visualize_results(input, prediction):
    """
    Visualization of model prediction (e.g. input and prediction).
    :param input: Image with only one channel (L); a black and white image. Dims of (32, 32, 1)
    :param prediction: Image with (a, b) channels; the color portion of the image. Dims of (32, 32, 2)
    :return: None
    """
    full_channel_img = np.dstack((input, prediction))
    as_rgb = color.lab2rgb(full_channel_img)
    plt.imshow(as_rgb)
    plt.show()


def run_model_and_plot_output(model, X, Y):
    predicted_ab_channels = model.call(X)
    for i in range(len(X)):
        visualize_results(X[i], predicted_ab_channels[i])
        as_rgb = color.lab2rgb(Y[i])
        plt.imshow(as_rgb)
        plt.show()


def main():
    print("Running preprocessing...")
    X_train, X_test, Y_train, Y_test = preprocess()
    print("Preprocessing complete.")

    only_printing_model_output = False

    model = ColorizationModel()
    checkpoint_dir = './tf_ckpts'
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print("Restoring model from latest checkpoint.")
        if only_printing_model_output:
            num_outputs_to_plot = 5
            input_batch, labels_batch = X_test[:num_outputs_to_plot], Y_test[:num_outputs_to_plot]
            run_model_and_plot_output(model, input_batch, labels_batch)
            return

    num_epochs = 10
    for epoch in range(num_epochs):
        train(model, X_train, Y_train)
        print(f"Training epoch {epoch} completed")
    save_path = manager.save()
    print(f"Saved model after latest training run at: {save_path}")

    acc = test(model, X_test, Y_test)
    print(f"Accuracy of Colorization model: {acc}")


if __name__ == '__main__':
    main()
