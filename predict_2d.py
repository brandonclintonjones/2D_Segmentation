###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########       AUGUST 4, 2020
#########
#########
#########
###############################
###############################
###############################
###############################




import tensorflow as tf
import numpy as np
from tensorflow import fft2d
from tensorflow import ifft2d
import timeit
from tensorflow.python import roll
import tqdm
import os
import glob
from tensorflow.keras.layers import Lambda


#############
#############
#############       CUSTOM DEPENDENCIES!!!!!
#############
#############

import model_architectures

# from model_architectures import unet_7_layers, simple_cnn
from my_image_generator import data_generator
from show_3d_images import show_3d_images

from unet_2d import CNN


def main():
    """
        Tests the CNN.

        """

    # CNN Parameters

    run_2d = True    
    batch_size = 8
    max_epoch = 40
    lr = 1e-5


    # GENERATOR PARAMETERS
    zoom_range = (1,1)
    max_rotation_degree = 0
    width_shift_ratio = 0
    height_shift_ratio = 0
    flipud_bool = False
    fliplr_bool = False



    the_generator = data_generator(
        bool_2d=run_2d,
        batch_size=batch_size,
        bool_shuffle = True,
        zoomRange=zoom_range, 
        rotationRange=max_rotation_degree, 
        widthShiftRange=width_shift_ratio, 
        heightShiftRange=height_shift_ratio,
        flipLR=fliplr_bool, 
        flipUD=flipud_bool
        )



    # if run_2d:
    #     the_generator.display_2d_image_masks()
    # else:
    #     the_generator.display_3d_image_masks()

    (
    img_height,
    img_width,
    scan_names_train,
    scan_names_valid
    ) = the_generator.pass_info()



    name = "b_{}_e_{}_lr_{}".format(
        str(batch_size),  str(max_epoch), str(lr)
    )

    convnet = CNN(
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
        learn_rate=lr,
        max_epoch=max_epoch,
        model_name = name,
    )


    convnet.load()

    num_scans = len(scan_names_train)

    num_steps = int(np.floor(num_scans/batch_size))




    for steps in range(num_steps):
        
        (batch_image,batch_mask)=the_generator.get_batch()

        mask_prediction = convnet.predict( X_star = batch_image)

        mask_prediction = scale_0_1(mask_prediction)


        # mask_prediction[mask_prediction>0.5]=1
        # mask_prediction[mask_prediction<0.5]=0

        mask_prediction = np.squeeze(mask_prediction).transpose((1,2,0))

        batch_image = np.squeeze(batch_image).transpose((1,2,0))
        batch_mask = np.squeeze(batch_mask).transpose((1,2,0))

        # show_3d_images(mask_prediction_thresh)


        mask_prediction_thresh = np.where(mask_prediction >= 0.7, 1.0, 0.0 )

        # mask_prediction_thresh = mask_prediction > 0.5
        

        print('MAX MIN PREDICTION')
        print(np.amax(mask_prediction))
        print(np.amin(mask_prediction))
        print('\n\n DSC ')
        dsc = dice_coef(y_true = batch_mask,y_pred = mask_prediction)
        print(dsc)
        print('\n\n')

        # print(aa)

        display_matrix = np.hstack((batch_image,batch_mask,mask_prediction,mask_prediction_thresh))

        show_3d_images(display_matrix)


        # for predictions in range(batch_size):
        #     image_input = batch_image[predictions,:]
        #     mask_label = batch_mask[predictions,:]

        #     mask_prediction = convnet.predict(X_star = image_input)

        #     mask_prediction[mask_prediction>0.5]=1
        #     mask_prediction[mask_prediction<0.5]=0

        #     print('SHAPES')
        #     print(mask_prediction.shape)
        #     print(image_input.shape)
        #     print(mask_label.shape)

        #     display_matrix = np.hstack((image_input,mask_label,mask_prediction))

        #     show_3d_images(display_matrix)





def dice_coef(y_true,y_pred,smooth=1):
    # Expects numpy inputs
    
    y_pred[y_pred>=0.5] = 1.0
    y_pred[y_pred<0.5] = 0.0


    y_true = y_true.flatten()
    y_pred = y_pred.flatten()


    intersection = np.multiply(y_true,y_pred).sum()
    union = y_pred.sum() + y_true.sum()
    dsc = (2*intersection ) / (union )
    
    return dsc


def scale_0_1(mask):
    mask = mask - np.amin(mask)
    mask = mask / np.amax(mask)
    return mask


if __name__ == "__main__":
    
    main()
