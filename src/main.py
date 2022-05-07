import tensorflow as tf

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

    # training model on batches 
    for i in range(num_batches):
        startindex = int(i * model.batch_size)
        endindex = int((i + 1) * model.batch_size)
        input_batch = X[startindex:endindex]
        labels_batch = Y[startindex:endindex]

        # make sure to use tf.stack to stack the grayscale and ab channels 
        with tf.GradientTape() as tape:
            probs = model.call(input_batch)
            probs = tf.stack(input_batch, probs)
            loss = model.loss_function(probs, labels_batch)

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


def visualize_results():
    """
    Visualization of model prediction (e.g. input and prediction).
    Probably easiest to use matplotlib.
    :return: None
    """
    pass


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
