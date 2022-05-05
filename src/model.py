import tensorflow as tf
from keras.initializers.initializers_v1 import TruncatedNormal
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D


class ColorizationModel(tf.keras.Model):
    def __init__(self):
        super(ColorizationModel, self).__init__()

        # TODO
        # 1) Define any hyperparameters, optimizer, ...

        #Hyperparameters
        self.learning_rate = 0.00001
        self.l_cent = 50
        self.l_norm = 100
        self.ab_norm = 110
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.stdev = 0.04

        # TODO
        # 2) Define layers
        kernel_init = TruncatedNormal(stddev=self.stdev)

        self.model = tf.keras.Sequential()

        #conv1 section
        self.model.add(Conv2D(filters=64, kernel_size=3,strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=2, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, ctivation='relu'))
        self.model.add(BatchNormalization())

        #conv2 section
        self.model.add(Conv2D(filters=128, kernel_size=3, strides=1, dilation_rate=1,padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=128, kernel_size=3, strides=2, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(BatchNormalization())

        #conv3 section
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init,activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=2, dilation_rate=1,padding="same",
                              kernel_initializer=kernel_init,activation='relu'))
        self.model.add(BatchNormalization())

        #conv4 section
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1,padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1,  padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='relu'))
        self.model.add(BatchNormalization())

        # conv5 section (dilation rate is 2
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1,dilation_rate=2,
                              kernel_initializer=kernel_init,padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2,
                              kernel_initializer=kernel_init,padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2,
                              kernel_initializer=kernel_init, padding="same", activation='relu'))
        self.model.add(BatchNormalization())

        # conv6 section (dilation rate is 2)
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1,dilation_rate=2,
                              kernel_initializer=kernel_init,padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2,
                              kernel_initializer=kernel_init,padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=2,
                              kernel_initializer=kernel_init, padding="same", activation='relu'))
        self.model.add(BatchNormalization())


        # conv7 section (dilation rate is 1)
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1,dilation_rate=1,
                              kernel_initializer=kernel_init,padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, dilation_rate=1,
                              kernel_initializer=kernel_init,padding="same", activation='relu'))
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
        
        # # I'm not sure if these are the correct layers?
        # self.model.add(Conv2D(filters=2, kernel_size=1, strides=1, padding="valid", bias=False))
        # self.model.add(UpSampling2D(size=(2, 2), interpolation='bilinear'))

        #conv9 section
        self.model.add(Conv2DTranspose(filters=256, kernel_size=4, strides=4, padding="SAME",
                                       kernel_initializer=kernel_init, activation='relu'))
        self.model.add(BatchNormalization())

        # y_hat
        self.model.add(Conv2D(filters=2, kernel_size=1, strides=1, dilation_rate=1, padding="same",
                              kernel_initializer=kernel_init, activation='softmax'))
        

    @tf.function
    def call(self, inputs):
        """
        :param : ...
        :return ...
        """
        return self.model(inputs)
        

    # I don't think we need an accuracy function here since this isn't a classification problem?
    def accuracy_function(self):
        """
        """
        # TODO
        pass

    def loss_function(self,predictions,labels):
        """
        """
        # TODO
        loss = self.mse(labels, predictions).numpy()
        
        num_imgs,h,w = predictions.shape[0],predictions.shape[1],predictions.shape[2]
