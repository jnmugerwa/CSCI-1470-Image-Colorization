import tensorflow as tf
from keras.initializers.initializers_v1 import TruncatedNormal
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization


class ColorizationModel(tf.keras.Model):
    def __init__(self):
        super(ColorizationModel, self).__init__()

        # 1) Define any hyperparameters, optimizer, ...
        self.learning_rate = 0.00001
        self.batch_size = 40  # we can test different batch sizes
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.stdev = 0.04

        # 2) Define layers
        kernel_init = TruncatedNormal(stddev=self.stdev)

        self.model = tf.keras.Sequential()

        # conv1 section
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=2, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(BatchNormalization())

        # conv2 section
        self.model.add(Conv2D(filters=128, kernel_size=3, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=128, kernel_size=3, strides=2, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(BatchNormalization())

        # conv3 section
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=2, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(BatchNormalization())

        # conv4 section
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(BatchNormalization())

        # conv5 section (dilation rate is 2
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2,
                              kernel_initializer=kernel_init, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2,
                              kernel_initializer=kernel_init, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2,
                              kernel_initializer=kernel_init, padding="same", activation='relu'))
        self.model.add(BatchNormalization())

        # conv6 section (dilation rate is 2)
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2,
                              kernel_initializer=kernel_init, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2,
                              kernel_initializer=kernel_init, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2,
                              kernel_initializer=kernel_init, padding="same", activation='relu'))
        self.model.add(BatchNormalization())

        # conv7 section (dilation rate is 1)
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1,
                              kernel_initializer=kernel_init, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1,
                              kernel_initializer=kernel_init, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1,
                              kernel_initializer=kernel_init, padding="same", activation='relu'))
        self.model.add(BatchNormalization())

        # conv8 section
        section_eight_kernel_init = TruncatedNormal(stddev=0.1)
        self.model.add(Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="SAME",
                                       kernel_initializer=section_eight_kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="SAME",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=4, strides=1, padding="SAME",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(BatchNormalization())

        # conv9 section
        self.model.add(Conv2DTranspose(filters=256, kernel_size=4, strides=4, padding="SAME",
                                       kernel_initializer=kernel_init, activation='relu'))
        self.model.add(BatchNormalization())

        # Removed the softmax activation because we no longer want probabilities
        self.model.add(Conv2D(filters=2, kernel_size=1, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init))

    @tf.function
    def call(self, inputs):
        """
        :param inputs: Black-and-white images, in L*a*b space. Dimensions of (batch_size, 32, 32, 1)
        :return The predicted (a, b) channels for each batch image. Dimensions of (batch_size, 32, 32, 2)
        """
        return self.model(inputs)

    def loss_function(self, predictions, labels):
        """
        :param predictions: The predicted values of the (a, b) channels of each batch image
        :param labels: The true values of the (a, b) channels of each batch image
        :return: the MSE of the values in the predicted vs true channels
        """
        return self.mse(labels, predictions)
