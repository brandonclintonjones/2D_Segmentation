###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########		AND CARLOS ADOLFO OSUNA
#########       APRIL 23, 2020
#########
#########
#########
###############################
###############################
###############################
###############################

from tensorflow.layers import Conv1D, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose
from tensorflow.layers import Dense, Dropout, Flatten, Layer
from tensorflow.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from tensorflow.layers import BatchNormalization


# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda, Reshape
# from tensorflow.keras.layers import Conv2D, UpSampling2D, SpatialDropout2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
# from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, SpatialDropout3D
# from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import concatenate
# from tensorflow.keras.layers import Permute
# from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import os
import numpy as np
import tensorflow as tf



# def conv_block_simple_3d(prevlayer, num_filters, prefix, kernel_size=(2,3,3),initializer="he_normal", strides=(1, 1, 1)):
def conv_block_simple_3d(prevlayer, num_filters, prefix, kernel_size=(2,3,3),initializer="glorot_normal", strides=(1, 1, 1)):
    
    conv = Conv3D(filters=num_filters, kernel_size=kernel_size, padding="same", kernel_initializer=initializer, strides=strides, name=prefix + "_conv",
         data_format='channels_first')(prevlayer)
         
    conv = BatchNormalization(name=prefix + "_bn",
        axis=1)(conv)

    # conv = Activation('relu', name=prefix + "_activation")(conv)
    conv = tf.nn.relu(conv,name=prefix + "_activation")

    return conv




# def conv_block_simple_3d(prevlayer, num_filters, prefix, kernel_size=(3,3,3),initializer="he_normal", strides=(1, 1, 1)):
def conv_block_simple_3d_no_bn(prevlayer, num_filters, prefix, kernel_size=(3,3,3),initializer="glorot_normal", strides=(1, 1, 1)):
    
    conv = Conv3D(filters=num_filters, kernel_size=kernel_size, padding="same", kernel_initializer=initializer, strides=strides, name=prefix + "_conv",
         data_format='channels_first')(prevlayer)

    conv = tf.nn.relu(conv,name=prefix + "_activation")
    # conv = Activation('relu', name=prefix + "_activation")(conv)

    return conv



def unet_7_layers_3D(input_tensor):


	# print('INPUT IMAGE SHAPE')
	# print(input_tensor.shape)
	
	mp_param = (1,2,2) # (1,2,2)
	stride_param=(1,2,2)
	d_format = "channels_first"
	pad = "same"
	us_param = (1,2,2)
	# kern=(1,3,3)
	kern=(2,3,3)
	
	# filt=(32,64,128,256,512)
	filt=(32,64,128,256)
	# filt=(64,128,256,512,1024)

	conv1 = conv_block_simple_3d(prevlayer=input_tensor, num_filters=filt[0], prefix="conv1",kernel_size=kern)
	conv1 = conv_block_simple_3d(prevlayer=conv1, num_filters=filt[0], prefix="conv1_1",kernel_size=kern)
	pool1 = MaxPooling3D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool1")(conv1)

	conv2 = conv_block_simple_3d(prevlayer=pool1, num_filters=filt[1], prefix="conv2",kernel_size=kern)
	conv2 = conv_block_simple_3d(prevlayer=conv2, num_filters=filt[1], prefix="conv2_1",kernel_size=kern)
	pool2 = MaxPooling3D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool2")(conv2)


	conv3 = conv_block_simple_3d(prevlayer=pool2, num_filters=filt[2], prefix="conv3",kernel_size=kern)
	conv3 = conv_block_simple_3d(prevlayer=conv3, num_filters=filt[2], prefix="conv3_1",kernel_size=kern)
	pool3 = MaxPooling3D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool3")(conv3)



	conv4 = conv_block_simple_3d(prevlayer=pool3, num_filters=filt[3], prefix="conv_4",kernel_size=kern)
	conv4 = conv_block_simple_3d(prevlayer=conv4, num_filters=filt[3], prefix="conv_4_1",kernel_size=kern)
	conv4 = conv_block_simple_3d(prevlayer=conv4, num_filters=filt[3], prefix="conv_4_2",kernel_size=kern)



	up5 = Conv3DTranspose(filters=filt[2],kernel_size=kern,strides=(1,2,2),padding="same",data_format="channels_first")(conv4)


	up5 = concatenate([up5, conv3], axis=1)
	conv5 = conv_block_simple_3d(prevlayer=up5, num_filters=filt[2], prefix="conv5_1")
	conv5 = conv_block_simple_3d(prevlayer=conv5, num_filters=filt[2], prefix="conv5_2")


	up6 = Conv3DTranspose(filters=filt[1],kernel_size=kern,strides=(1,2,2),padding="same",data_format="channels_first")(conv5)


	up6 = concatenate([up6, conv2], axis=1)
	conv6 = conv_block_simple_3d(prevlayer=up6, num_filters=filt[1], prefix="conv6_1")
	conv6 = conv_block_simple_3d(prevlayer=conv6, num_filters=filt[1], prefix="conv6_2")


	up7 = Conv3DTranspose(filters=filt[0],kernel_size=kern,strides=(1,2,2),padding="same",data_format="channels_first")(conv6)
	up7 = concatenate([up7, conv1], axis=1)
	conv7 = conv_block_simple_3d(prevlayer=up7, num_filters=filt[0], prefix="conv7_1")
	conv7 = conv_block_simple_3d(prevlayer=conv7, num_filters=filt[0], prefix="conv7_2")

	# conv9 = SpatialDropout2D(0.2,data_format=d_format)(conv9)

	prediction = Conv3D(filters=1, kernel_size=(1, 1, 1), activation="sigmoid", name="prediction", data_format=d_format)(conv7)


	# print('PREDICTION SHAPE')
	# print(prediction.shape)

	return prediction


def unet_9_layers_3D(input_shape):

	img_input = Input(input_shape)

	# print('INPUT IMAGE SHAPE')
	# print(img_input.shape)
	
	mp_param = (1,2,2) # (1,2,2)
	stride_param=(1,2,2)
	d_format = "channels_first"
	pad = "same"
	us_param = (1,2,2)

	filt=(32,64,128,256,512)
	# filt=(64,128,256,512,1024)

	conv1 = conv_block_simple_3d(prevlayer=img_input, num_filters=filt[0], prefix="conv1")
	conv1 = conv_block_simple_3d(prevlayer=conv1, num_filters=filt[0], prefix="conv1_1")
	pool1 = MaxPooling3D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool1")(conv1)

	conv2 = conv_block_simple_3d(prevlayer=pool1, num_filters=filt[1], prefix="conv2")
	conv2 = conv_block_simple_3d(prevlayer=conv2, num_filters=filt[1], prefix="conv2_1")
	pool2 = MaxPooling3D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool2")(conv2)


	conv3 = conv_block_simple_3d(prevlayer=pool2, num_filters=filt[2], prefix="conv3")
	conv3 = conv_block_simple_3d(prevlayer=conv3, num_filters=filt[2], prefix="conv3_1")
	pool3 = MaxPooling3D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool3")(conv3)


	conv4 = conv_block_simple_3d(prevlayer=pool3, num_filters=filt[3], prefix="conv4")
	conv4 = conv_block_simple_3d(prevlayer=conv4, num_filters=filt[3], prefix="conv4_1")
	pool4 = MaxPooling3D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool4")(conv4)


	conv5 = conv_block_simple_3d(prevlayer=pool4, num_filters=filt[4], prefix="conv_5")
	conv5 = conv_block_simple_3d(prevlayer=conv5, num_filters=filt[4], prefix="conv_5_1")
	conv5 = conv_block_simple_3d(prevlayer=conv5, num_filters=filt[4], prefix="conv_5_2")


	up6 = UpSampling3D(size=us_param,data_format=d_format)(conv5)

	up6 = concatenate([up6, conv4], axis=1)
	conv6 = conv_block_simple_3d(prevlayer=up6, num_filters=filt[3], prefix="conv6_1")
	conv6 = conv_block_simple_3d(prevlayer=conv6, num_filters=filt[3], prefix="conv6_2")


	up7 = UpSampling3D(size=us_param,data_format=d_format)(conv6)
	up7 = concatenate([up7, conv3], axis=1)
	conv7 = conv_block_simple_3d(prevlayer=up7, num_filters=filt[2], prefix="conv7_1")
	conv7 = conv_block_simple_3d(prevlayer=conv7, num_filters=filt[2], prefix="conv7_2")


	up8 = UpSampling3D(size=us_param,data_format=d_format)(conv7)
	up8 = concatenate([up8, conv2], axis=1)
	conv8 = conv_block_simple_3d(prevlayer=up8, num_filters=filt[1], prefix="conv8_1")
	conv8 = conv_block_simple_3d(prevlayer=conv8, num_filters=filt[1], prefix="conv8_2")


	up9 = UpSampling3D(size=us_param,data_format=d_format)(conv8)
	up9 = concatenate([up9, conv1], axis=1)
	conv9 = conv_block_simple_3d(prevlayer=up9, num_filters=filt[0], prefix="conv9_1")
	conv9 = conv_block_simple_3d(prevlayer=conv9, num_filters=filt[0], prefix="conv9_2")

	# conv9 = SpatialDropout2D(0.2,data_format=d_format)(conv9)

	prediction = Conv3D(filters=1, kernel_size=(1, 1, 1), activation="sigmoid", name="prediction", data_format=d_format)(conv9)
	model = Model(img_input, prediction)

	# print('PREDICTION SHAPE')
	# print(prediction.shape)

	return model







def unet_9_layers(input_tensor,output_tensor_channels = 1):

	mp_param = (2,2)
	stride_param=(2,2)
	d_format = "channels_first"
	pad = "same"
	kern=(3,3)


	filt=(32,64,128,256,512)
	# filt=(64,128,256,512,1024)

	conv1 = conv_block_simple_2d(prevlayer=input_tensor, num_filters=filt[0], prefix="conv1")
	conv1 = conv_block_simple_2d(prevlayer=conv1, num_filters=filt[0], prefix="conv1_1")
	pool1 = MaxPooling2D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool1")(conv1)

	conv2 = conv_block_simple_2d(prevlayer=pool1, num_filters=filt[1], prefix="conv2")
	conv2 = conv_block_simple_2d(prevlayer=conv2, num_filters=filt[1], prefix="conv2_1")
	pool2 = MaxPooling2D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool2")(conv2)


	conv3 = conv_block_simple_2d(prevlayer=pool2, num_filters=filt[2], prefix="conv3")
	conv3 = conv_block_simple_2d(prevlayer=conv3, num_filters=filt[2], prefix="conv3_1")
	pool3 = MaxPooling2D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool3")(conv3)


	conv4 = conv_block_simple_2d(prevlayer=pool3, num_filters=filt[3], prefix="conv4")
	conv4 = conv_block_simple_2d(prevlayer=conv4, num_filters=filt[3], prefix="conv4_1")
	pool4 = MaxPooling2D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool4")(conv4)

	conv5 = conv_block_simple_2d(prevlayer=pool4, num_filters=filt[4], prefix="conv_5")
	conv5 = conv_block_simple_2d(prevlayer=conv5, num_filters=filt[3], prefix="conv_5_1")
	# conv5 = conv_block_simple_2d(prevlayer=conv5, num_filters=filt[4], prefix="conv_5_2")

	# 4 is 512,
	# 3 is 256
	# 2 is 128

	up6 = Conv2DTranspose(filters=filt[3],kernel_size=kern,strides=(2,2),padding="same",data_format="channels_first")(conv5)
	up6 = concatenate([up6, conv4], axis=1)

	conv6 = conv_block_simple_2d(prevlayer=up6, num_filters=filt[3], prefix="conv6_1")
	conv6 = conv_block_simple_2d(prevlayer=conv6, num_filters=filt[2], prefix="conv6_2")



	up7 = Conv2DTranspose(filters=filt[2],kernel_size=kern,strides=(2,2),padding="same",data_format="channels_first")(conv6)
	up7 = concatenate([up7, conv3], axis=1)
	conv7 = conv_block_simple_2d(prevlayer=up7, num_filters=filt[2], prefix="conv7_1")
	conv7 = conv_block_simple_2d(prevlayer=conv7, num_filters=filt[1], prefix="conv7_2")



	up8 = Conv2DTranspose(filters=filt[1],kernel_size=kern,strides=(2,2),padding="same",data_format="channels_first")(conv7)
	up8 = concatenate([up8, conv2], axis=1)
	conv8 = conv_block_simple_2d(prevlayer=up8, num_filters=filt[1], prefix="conv8_1")
	conv8 = conv_block_simple_2d(prevlayer=conv8, num_filters=filt[0], prefix="conv8_2")



	up9 = Conv2DTranspose(filters=filt[0],kernel_size=kern,strides=(2,2),padding="same",data_format="channels_first")(conv8)
	up9 = concatenate([up9, conv1], axis=1)
	conv9 = conv_block_simple_2d(prevlayer=up9, num_filters=filt[0], prefix="conv9_1")
	conv9 = conv_block_simple_2d(prevlayer=conv9, num_filters=filt[0], prefix="conv9_2")

	# conv9 = SpatialDropout2D(0.2,data_format=d_format)(conv9)

	# prediction = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", name="prediction", data_format=d_format)(conv9)

	prediction = Conv2D(filters = output_tensor_channels, kernel_size = (1, 1), activation="sigmoid", use_bias = True,
		name="prediction",data_format=d_format)(conv9)
	# prediction = Conv2D(filters = output_tensor_channels, kernel_size = (1, 1), name="prediction",data_format=d_format)(conv9)

	return prediction


def unet_7_layers(input_tensor):


	# print('INPUT IMAGE SHAPE')
	# print(input_tensor.shape)
	
	mp_param = (2,2) # (1,2,2)
	stride_param=(2,2)
	d_format = "channels_first"
	pad = "same"	
	us_param = (2,2)
	kern=(3,3)

	# filt=(32,64,128,256,512)
	filt=(32,64,128,256)
	# filt=(64,128,256,512,1024)



	conv1 = conv_block_simple_2d(prevlayer=input_tensor, num_filters=filt[0], prefix="conv1",kernel_size=kern)
	conv1 = conv_block_simple_2d(prevlayer=conv1, num_filters=filt[0], prefix="conv1_1",kernel_size=kern)
	pool1 = MaxPooling2D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool1")(conv1)

	conv2 = conv_block_simple_2d(prevlayer=pool1, num_filters=filt[1], prefix="conv2",kernel_size=kern)
	conv2 = conv_block_simple_2d(prevlayer=conv2, num_filters=filt[1], prefix="conv2_1",kernel_size=kern)
	pool2 = MaxPooling2D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool2")(conv2)


	conv3 = conv_block_simple_2d(prevlayer=pool2, num_filters=filt[2], prefix="conv3",kernel_size=kern)
	conv3 = conv_block_simple_2d(prevlayer=conv3, num_filters=filt[2], prefix="conv3_1",kernel_size=kern)
	pool3 = MaxPooling2D(pool_size=mp_param,strides=stride_param,
		padding="same",data_format="channels_first",name="pool3")(conv3)



	conv4 = conv_block_simple_2d(prevlayer=pool3, num_filters=filt[3], prefix="conv_4",kernel_size=kern)
	conv4 = conv_block_simple_2d(prevlayer=conv4, num_filters=filt[3], prefix="conv_4_1",kernel_size=kern)
	conv4 = conv_block_simple_2d(prevlayer=conv4, num_filters=filt[3], prefix="conv_4_2",kernel_size=kern)



	up5 = Conv2DTranspose(filters=filt[2],kernel_size=kern,strides=(2,2),padding="same",data_format="channels_first")(conv4)


	up5 = concatenate([up5, conv3], axis=1)
	conv5 = conv_block_simple_2d(prevlayer=up5, num_filters=filt[2], prefix="conv5_1")
	conv5 = conv_block_simple_2d(prevlayer=conv5, num_filters=filt[2], prefix="conv5_2")


	up6 = Conv2DTranspose(filters=filt[1],kernel_size=kern,strides=(2,2),padding="same",data_format="channels_first")(conv5)


	up6 = concatenate([up6, conv2], axis=1)
	conv6 = conv_block_simple_2d(prevlayer=up6, num_filters=filt[1], prefix="conv6_1")
	conv6 = conv_block_simple_2d(prevlayer=conv6, num_filters=filt[1], prefix="conv6_2")


	up7 = Conv2DTranspose(filters=filt[0],kernel_size=kern,strides=(2,2),padding="same",data_format="channels_first")(conv6)
	up7 = concatenate([up7, conv1], axis=1)
	conv7 = conv_block_simple_2d(prevlayer=up7, num_filters=filt[0], prefix="conv7_1")
	conv7 = conv_block_simple_2d(prevlayer=conv7, num_filters=filt[0], prefix="conv7_2")

	# conv9 = SpatialDropout2D(0.2,data_format=d_format)(conv9)

	prediction = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", name="prediction", data_format=d_format)(conv7)


	# print('PREDICTION SHAPE')
	# print(prediction.shape)

	return prediction









def simple_cnn(input_shape):
    # NOTE: ThE INPUT_TENSOR IS THE SUBSAMPLED KSPACE
    # SO IT IS A COMPLEX128 TENSOR OF SHAPE
    # ( BATCH_DIM, CHANNELS_DIM, IMG_HEIGHT, IMG_WIDTH )

    img_input = Input(input_shape) 

    conv1 = conv_block_simple_2d(prevlayer=img_input, num_filters=16, prefix="conv1")
    conv2 = conv_block_simple_2d(prevlayer=conv1, num_filters=16, prefix="conv1_2")
    conv3 = conv_block_simple_2d(prevlayer=conv2, num_filters=16, prefix="conv1_3")
    prediction = Conv2D(
        filters=2, 
        kernel_size=(1, 1),
        activation="sigmoid",
        name="prediction",
        data_format="channels_first",
    )(conv3)
    # prediction = Conv2D(
    #     filters=1, 
    #     kernel_size=(1, 1),
    #     activation="sigmoid",
    #     name="prediction",
    #     data_format="channels_first",
    # )(conv3)
    the_model=Model(inputs=img_input,outputs=prediction)

    return the_model




def conv_block_simple_2d(prevlayer, num_filters, prefix, kernel_size=(3,3),initializer="he_normal", strides=(1, 1)):
    
    # conv = Conv2D(filters=num_filters, kernel_size=kernel_size, padding="same", kernel_initializer=initializer, strides=strides, name=prefix + "_conv",
    #      data_format='channels_first')(prevlayer)
         
    # conv = BatchNormalization(name=prefix + "_bn",
    #     axis=1)(conv)

    # conv = Activation('relu', name=prefix + "_activation")(conv)

    # return conv
    conv = Conv2D(filters=num_filters, kernel_size=kernel_size, padding="same", kernel_initializer=initializer, strides=strides, name=prefix + "_conv",
         data_format='channels_first')(prevlayer) 
         
    conv = BatchNormalization(name=prefix + "_bn",
        axis=1)(conv)

    # conv = Activation('relu', name=prefix + "_activation")(conv)
    conv = tf.nn.relu(conv,name=prefix + "_activation")

    return conv


def conv_block_simple_2d(prevlayer, num_filters, prefix, kernel_size=(3,3),initializer="he_normal", strides=(1, 1)):
    
    # conv = Conv2D(filters=num_filters, kernel_size=kernel_size, padding="same", kernel_initializer=initializer, strides=strides, name=prefix + "_conv",
    #      data_format='channels_first')(prevlayer)

    # conv = Activation('relu', name=prefix + "_activation")(conv)

    # return conv

    conv = Conv2D(filters=num_filters, kernel_size=kernel_size, padding="same", kernel_initializer=initializer, strides=strides, name=prefix + "_conv",
         data_format='channels_first')(prevlayer)

         
    conv = BatchNormalization(name=prefix + "_bn",
        axis=1)(conv)

    # conv = Activation('relu', name=prefix + "_activation")(conv)
    conv = tf.nn.relu(conv,name=prefix + "_activation")

    return conv

