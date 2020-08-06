###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########       APRIL 22, 2020
#########
#########
#########
###############################
###############################
###############################
###############################

import argparse
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.signal.fft_ops import fft2d, ifft2d, fftshift, ifftshift, fft3d, ifft3d

# from tensorflow.signal import fft2d, ifft2d, fftshift, ifftshift
from tensorflow.compat.v1 import Session, placeholder

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda, Reshape
from tensorflow.keras.layers import (
    Conv2D,
    UpSampling2D,
    SpatialDropout2D,
    MaxPooling2D,
    ZeroPadding2D,
    Conv2DTranspose,
)
from tensorflow.keras.layers import (
    Conv3D,
    MaxPooling3D,
    UpSampling3D,
    Activation,
    BatchNormalization,
    PReLU,
    SpatialDropout3D,
)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Permute
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

#############
#############
#############       CUSTOM DEPENDENCIES!!!!!
#############
#############

from data_loader_class import data_generator


class CNN(tf.Module):
    def __init__(self, input_shape, bool_2d=True ,*layerinfo):
        """
            Defines the CNN structure

            Parameters:
                layerinfo = list of tuples containing a tf.layers function
                    and its parameters in kwargs format
                input_shape = expected image shape for inputs
                # NOTE: input_shape is for a given batch
                # 
                # input_shape = ( num_channels, img_height, img_width, num_slices )

            Outputs:
                image = CNN-processed image 
            """
        self.layerinfo = layerinfo

        self.training_loss = []
        self.valid_loss = []

        self.sess = Session()
        # physical_devices = tf.config.experimental.list_logical_devices('GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0],True)

        self.learn_rate = 1e-3
        self.max_epoch = 2

        self.bool_2d = bool_2d


    def forward_pass(self, input):
        """
            Runs the image through the layer structure.

            """

        if issubclass(type(input), tf.Tensor):
            # If the input is not a tensor, then we switch from Numpy to a Tensor object
            # And define boolean so we reverse the operation if needed
            casted = True
            original_class = type(input)
            # input = tf.Tensor(input)
            input_tf = tf.convert_to_tensor(input, dtype=tf.complex128)

        else:
            casted = False

        input = the_nn(input)

        # for layer, parameters in self.layerinfo:
        #     input = layer(input, **parameters)
        #     # Call a tensorflow layer with specified parameters
        #     # Example:
        #     #       layer( tf.layers.Conv3D ("kernel_size": 6, "actiavation": tf.nn.relu)

        if casted:
            input = original_class(input)

        return input


    def train(self, inputs, expectations, epochs, batchsize=None, *args, **kwargs):
        """
            Call syntax:
                model.train(inputs,expectations,epochs = 1e3)
                NOTE: inputs is subsampled kspace
                NOTE: expectations is fully sampled kspace or image?
            
            Takes in the following training parameter arguments:
                epochs:int = # of iterations to optimize the loss function
                batchsize:int = size of subset of data to use for an optimizing epoch
                        - Default: None [no batching]

            Can take in the following *args:
                None
            
            Can take in the following **kwargs:
                loss = loss function to be optimized which takes in the following arguments:
                        - loss(self,inputs,expectations,*lossargs,**losskwargs)
                        where lossargs and losskwargs can be defined as additional **kwargs.
                        - Default: tf.nn.l2_loss [decorated to accept inputs in expected format]
                optimizer = optimizer function to perform optimization on loss function, taking on:
                        - optimizer(self.parameters(),*optimizerargs,**optimizerkwargs)
                        where optimizerargs and optimizerkwargs can be defined as additional **kwargs.
                        - Default: tf.compat.v1.train.AdamOptimizer()
                cutoff:callable = cutoff function to truncate optimization, taking:
                        - cutoff(self,lossoutputs,iter,*cutoffargs,**cutoffkwargs)
                        where lossoutputs is a numpy.ndarray of the history of loss evaluations
                        and iter is the iteration #
                """

        ##############
        ##############
        ############## REPLACE THIS WITH DATA LOADER HERE
        ##############
        ##############

        # inputs, expectations, kwargs = updatedefaults(inputs, expectations, kwargs)

        # NOTE: THE CODE BELOW IS THE ACTUAL TRAINING LOOP ITERATION
        # Will need to be modified to work for tensorflow

        my_generator = data_generator(bool_2d=self.bool_2d)

        (
            train_images,
            train_kspace,
            train_names,
            train_kspace_undersampled,
            valid_images,
            valid_kspace,
            valid_names,
            valid_kspace_undersampled,
        ) = my_generator.get_batch_tf()

        # NOTE: 2D TRAIN_IMAGES HAS THE FORM
        #       (BATCH_DIM, IMG_HEIGHT, IMG_WIDTH)
        # 3D TRAIN_IMAGES IS (BATCH_DIM, IMG_HEIGHT, IMG_WIDTH, NUM_SCANS)
        
        num_train = num_channels = img_height = img_width = 0

        if self.bool_2d:

          (num_train, num_channels, img_height, img_width) = train_images.shape
          input_shape = (None, num_channels, img_height, img_width)

        else:
          (num_train, num_channels, img_height, img_width, num_slices) = train_images.shape

          input_shape = (None, num_channels, img_height, img_width, num_slices)

        model = simple_cnn(input_shape)

        model.summary()

        model.compile(
            optimizer=Adam(self.learn_rate),
            loss=tf.keras.losses.MSE,
            metrics=["MSE"],
            )


        for epoch_num in range(self.max_epoch):

            model.train_on_batch(x=train_kspace_undersampled,y=train_kspace)

            # Also have
            # model.test_on_batch()
            # model.predict_on_batch()

        # results = model.fit()



        # # (None, Batch_Dim, Channel_Dimension, Image_dims* )
        # self.kspace_subsampled_input_placeholder = placeholder(tf.complex128, shape=(None,1,img_height,img_width) )
        #     # NOTE: UNDERSAMPLED KSPACE PLACEHOLDER
        #     # DEFINES DIMESNIONS OF INPUT UNDERSAMPLED KSPACE

        # self.kspace_fully_sampled_placeholder = placeholder(tf.complex128, shape=(None,1,img_height,img_width ) )
        # #

        # self.kspace_cnn_predicted = self.forward_pass(self.kspace_undersampled_input_placeholder)
        # # NOTE: THIS IS THE OUTPUT OF THE MODEL
        # #   Also called Y_PRED

        # self.loss = tf.keras.losses.MSE(y_true=self.kspace_fully_sampled_placeholder,y_pred=self.kspace_cnn_predicted)

        # self.adam_optimizer = tf.train.AdamOptimizer(self.learn_rate)

        # self.optimizer = self.adam_optimizer.minimize(self.loss)

        # init = tf.global_variables_initializer()

        # self.sess.run(init)

        # tf_dict_train = {self.kspace_subsampled_input_placeholder: train_kspace_undersampled,
        # self.kspace_fully_sampled_placeholder: train_kspace}
        # # NOTE: IF NOT DOING FULL BATCH I WILL HAVE TO PUT THIS IN THE LOOPS

        # lossrecord = []

        # for counter in range(self.max_epoch):

        #     training_loss_value = self.sess.run( self.optimizer, tf_dict_train)

        #     self.lossrecord.append(training_loss_value)

        #     print ('Epoch [%d/%d], Loss: %.4f, Time: %2fs'
        #             %(epoch+1, num_epochs, training_loss_value, elapsed))

    def image_to_kspace_tf(image):
        (num_scans, num_channels, img_height, img_width) = image.shape
        # NOTE: MIGHT HAVE TO CONVERT IMAGE TO COMPLEX DOUBLE HERE
        kspace = tf.zeros(image.shape, dtype=tf.complex128)

        # (Null, batch_dim, channel_dim, height, width)
        # kspace = ifftshift( fft2d( fftshift( image )))
        
        # QUESTION: Why are we looping over the 2nd index?
        for counter in range(num_scans):
            kspace[:, counter, :, :, :] = ifftshift(
                fft2d(fftshift(image[:, counter, :, :, :]))
            )

        return kspace

    def image_to_kspace_tf_3d(image):
        (num_scans, num_channels, img_height, img_width, num_slices) = image.shape
        kspace = tf.zeros(image.shape, dtype=tf.complex128)

        for counter in range(num_scans):
            kspace[:, counter, :, :, :, :] = ifftshift(
                fft3d(fftshift(image[:, counter, :, :, :, :]))
            )
        return kspace

    def kspace_to_image_tf(kspace):
        (num_scans, num_channels, img_height, img_width) = image.shape
        image = tf.zeros(image.shape, dtype=tf.complex128)

        # (Null, batch_dim, channel_dim, height, width)
        # kspace = ifftshift( fft2d( fftshift( image )))

        for counter in range(num_scans):
            image[:, counter, :, :, :] = ifftshift(
                ifft2d(fftshift(kspace[:, counter, :, :, :]))
            )

        # image = tf.math.abs( image )

        return image

    def kspace_to_image_tf_3d(kspace):
        (num_scans, num_channels, img_height, img_width, num_slices) = kspace.shape
        image = tf.zeros(kspace.shape, dtype=tf.complex128)

        for counter in range(num_scans):
            image[:, counter, :, :, :, :] = ifftshift(
                fft3d(fftshift(kspace[:, counter, :, :, :, :]))
            )
        return image
      



def simple_cnn(input_shape):
    # NOTE: ThE INPUT_TENSOR IS THE SUBSAMPLED KSPACE
    # SO IT IS A COMPLEX128 TENSOR OF SHAPE
    # ( BATCH_DIM, CHANNELS_DIM, IMG_HEIGHT, IMG_WIDTH )

    img_input = Input(input_shape) 

    conv1 = conv_block_simple_2d(prevlayer=input_tensor, num_filters=32, prefix="conv1")
    conv2 = conv_block_simple_2d(prevlayer=conv1, num_filters=32, prefix="conv1_1")
    conv3 = conv_block_simple_2d(prevlayer=conv2, num_filters=32, prefix="conv1_1")
    prediction = Conv2D(
        1,
        (1, 1, 1),
        activation="sigmoid",
        name="prediction",
        data_format="channels_first",
    )(conv3)


    the_model=Model(inputs=img_input,outputs=prediction)

    return the_model


# NOTE: COuld also do "he_normal" for convolution kernel initializer



def main():
    parser = argparse.ArgumentParser(description='Please specify if you would like to use the center 2D slice or whole 3D volume for each scan')
    parser.add_argument('--2d', dest='run_2d', action='store_true')
    parser.add_argument('--3d', dest='run_2d', action='store_false')
    parser.set_defaults(run_2d=True)

    args = parser.parse_args()

    run_2d = args.run_2d

    """
        Tests the CNN.

        """

    model = CNN(
        input_shape=(50, 50, 50, 1),
        bool_2d=run_2d
        # (
        #     tf.layers.conv3d,
        #     {
        #         "kernel_size": 5,
        #         "filters": 6,
        #         "strides": 1,
        #         "padding": "same",
        #         "activation": tf.nn.relu,
        #     },
        # ),
        # (
        #     tf.layers.dense,
        #     {
        #         "units": 120,
        #         "activation": tf.nn.relu,
        #     },
        # ),
    )


if __name__ == "__main__":
    main()
