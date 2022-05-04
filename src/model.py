import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D


class ColorizationModel(tf.keras.Model):
    def __init__(self):
        super(ColorizationModel, self).__init__()

        # TODO
        # 1) Define any hyperparameters, optimizer, ...
        
        self.l_cent = 50
        self.l_norm = 100
        self.ab_norm = 110

        # TODO
        # 2) Define layers
        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=128, kernel_size=3, strides=2, padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=2, padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu'))
        self.model.add(Conv2D(filters=313, kernel_size=1, strides=1, padding="valid", activation='softmax'))
        
        self.model.add(Conv2D(filters=2, kernel_size=1, strides=1, padding="valid", bias=False))
        self.model.add(UpSampling2D(size=(2, 2), interpolation='bilinear'))
        

    @tf.function
    def call(self, inputs):
        """
        :param : ...
        :return ...
        """
        # TODO
        return self.unnormalize_ab(self.model(inputs))
        

    def accuracy_function(self):
        """
        """
        # TODO
        pass

    def loss_function(self):
        """
        """
        # TODO
        pass
    
    def normalize_l(self, in_l):
        return (in_l-self.l_cent)/self.l_norm

    def unnormalize_l(self, in_l):
        return in_l*self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab/self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab*self.ab_norm
