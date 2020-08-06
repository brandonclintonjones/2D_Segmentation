###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########       august 4, 2020
#########
#########
#########
###############################
###############################
###############################
###############################

import numpy as np
from matplotlib import pyplot as plt
import pydicom
import glob
from scipy.io import loadmat
import numpy.fft as fft
import numpy.matlib
import os
import tensorflow as tf
import argparse
import logging

import scipy
# import pydicom
import cv2

#############
#############
#############       CUSTOM DEPENDENCIES!!!!!
#############
#############


from show_3d_images import show_3d_images



def get_logger(name):
    log_format = "%(asctime)s %(name)s %(levelname)5s %(message)s"
    logging.basicConfig(level=logging.DEBUG,format=log_format,
                        filename='dev.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(console)
    logging.basicConfig(filename='LOGFILE.log',filemode='w')

    return logging.getLogger(name)



class data_generator:
    def __init__(
        self,
        train_split = 0.8,
        bool_2d = True,
        batch_size = 5,
        bool_shuffle = True,
        zoomRange=(1,1), 
        rotationRange=0, 
        widthShiftRange=0, 
        heightShiftRange=0,
        flipLR=False, 
        flipUD=False
            ):

        super(data_generator, self).__init__()

        self.study_dir = os.getcwd()
        self.data_dir = self.study_dir + '/3D_Deidentified_data'
        self.is_2d = bool_2d

        print('\n\n\n')
        print(self.study_dir)
        print('\n\n\n')

        self.bool_shuffle = bool_shuffle
        self.batch_size = batch_size
        self.train_split = train_split

        seed = 2
        self.random = np.random.RandomState(seed=seed)
        self.zoomRange = zoomRange
        self.rotationRange = rotationRange
        self.widthShiftRange = widthShiftRange
        self.heightShiftRange = -heightShiftRange
        self.flipLR = flipLR
        self.flipUD = flipUD



        self.get_names_and_dimensions()


    def get_names_and_dimensions(self):
        # print('LOADING MAT NAMES . . . \n\n')

        data_dir = self.data_dir
        study_dir = self.study_dir


        scan_names = glob.glob(data_dir+'/img*')

        num_scans = len(scan_names)

        self.num_scans = num_scans


        # RANDOMIZE THE SCAN ORDER
        shuffle_arange = np.arange(num_scans)
        np.random.shuffle(shuffle_arange)

        tmp_names = scan_names
        for name_iter in range(num_scans):
            scan_names[name_iter]=tmp_names[shuffle_arange[name_iter]]

        # LOAD ONE IMAGE TO SAVE IMAGE DIMENSIONS
        img = loadmat(scan_names[0]+'/image.mat')
        img = img["image"] # NOTE: matfile has "D" key not "img"

        # img = img / np.amax(img)

        
        (img_height, img_width, num_slices) = img.shape

        self.dim = (img_height,img_width)
        self.img_height = img_height
        self.img_width = img_width
        self.num_slices = num_slices



        # SPLIT THE SCANS TO TRAIN AND VALID

        num_train = np.floor(num_scans*self.train_split).astype(int)
        num_valid = num_scans - num_train


        self.train_scan_names = scan_names[:num_train]

        self.valid_scan_names = scan_names[num_train:]


    def shuffle_data(self):
        # NOTE: ONLY SHUFFLE THE TRAINING DATA FOR GRADIENT DESCENT
        # NEVER A REASON TO SHUFFLE THE VALIDATION DATA

        num_train_scans = len(self.train_scan_names)
        

        total_arange = np.arange(num_train_scans)


        np.random.shuffle(total_arange)
        total_arange = total_arange.astype(np.int16)

        old_train_names = self.train_scan_names

        for counter in range(num_train_scans):
            new_index = total_arange[counter]
            self.train_scan_names[counter]=old_train_names[new_index]



    def load_files(self,file_names):
        # print('LOADING DATA FROM MAT FILES . . . \n\n')
        # print(file_names)

        num_scans_to_load = len(file_names)

        num_scans = len(file_names)
        img_height = self.img_height
        img_width = self.img_width
        num_slices = self.num_slices


        if self.is_2d:

            img_stack = np.zeros(
                (num_scans, 1, img_height, img_width), dtype=np.double
            )

            periosteal_stack = np.zeros(
                (num_scans, 1, img_height, img_width), dtype=np.double
            )


        else:

            img_stack = np.zeros(
                (num_scans, 1, img_height, img_width, num_slices), dtype=np.double
            )
            
            periosteal_stack = np.zeros(
                (num_scans, 1, img_height, img_width, num_slices), dtype=np.double
            )



        for counter in range(num_scans_to_load):

            img = loadmat(file_names[counter]+'/image.mat')
            img = img["image"] # NOTE: matfile has "D" key not "img"

            # show_3d_images(img)

            periosteal = loadmat(file_names[counter]+'/periosteal.mat')
            periosteal = periosteal["periosteal"] # NOTE: matfile has "D" key not "img"

            # show_3d_images(periosteal)

            periosteal = 1.0 - periosteal


            if self.is_2d:
                img_slice = img[:,:,29]
                img_slice = img_slice / np.amax(img_slice)

                img_stack[counter, 0, :] = img_slice
                periosteal_stack[counter, 0, :] = periosteal[:, :, 29]

            else:
                img = img/np.amax(img)

                img_stack[counter, 0, :,] = img
                periosteal_stack[counter, 0, :] = periosteal
                

        # print('\n\n DONE LOADING DATA!!!!! . . . \n\n\n ')

        

        return img_stack, periosteal_stack





    def get_batch(self,batch_ind = 0,is_train=True):


        if batch_ind == 0 and self.bool_shuffle:
            # print('SHUFFLING \n\n')
            self.shuffle_data()

        if is_train:
            scan_name_list = self.train_scan_names

        else:
            scan_name_list = self.valid_scan_names


        files_to_load = scan_name_list[batch_ind:(batch_ind+self.batch_size)]


        (images,periosteal_masks) = self.load_files(file_names=files_to_load)


        # periosteal_masks = self.periosteal_stack
        # images = self.img_stack

        # images = images[batch_ind:(batch_ind+self.batch_size)]
        # periosteal_masks = periosteal_masks[batch_ind:(batch_ind+self.batch_size)]




        images_augmented = np.empty(images.shape)
        periosteal_masks_augmented = np.empty(periosteal_masks.shape)

        # print('\n\n\n AUGMENTING DATA . . . \n\n\n ')


        for counter in range(self.batch_size):


            tmp_img = np.squeeze(images[counter,:])
            tmp_mask = np.squeeze(periosteal_masks[counter,:])

            (tmp_img,tmp_mask) = self.apply_augmentation(tmp_img,tmp_mask)

            images_augmented[counter,0,:]=tmp_img
            periosteal_masks_augmented[counter,0,:]=tmp_mask



        return (images_augmented , periosteal_masks_augmented)


    def apply_augmentation(self,img,mask):

        zf = self.getRandomZoomConfig(self.zoomRange)
        theta = self.getRandomRotation(self.rotationRange)
        tx, ty = self.getRandomShift(*self.dim, self.widthShiftRange, self.heightShiftRange)

        if self.flipLR:
            flipStackLR = self.getRandomFlipFlag()
        else:
            flipStackLR = False

        if self.flipUD:
            flipStackUD = self.getRandomFlipFlag()
        else:
            flipStackUD = False

        if self.is_2d:            


            # self.print_range(img)

            mask = self.applyZoom( mask , zf , 1 )
            img = self.applyZoom( img , zf , 0 )

            # self.print_range(img)

            mask = self.applyRotation( mask , theta , 1 )
            img = self.applyRotation( img , theta , 0 )

            # self.print_range(img)

            mask = self.applyShift( mask , tx, ty , 1 )
            img = self.applyShift( img , tx, ty , 0 )

            # self.print_range(img)


            if flipStackLR:
                img = np.fliplr( img )
                mask = np.fliplr( mask )

            if flipStackUD:

                img = np.flipud( img )
                mask = np.flipud( mask )



        else:

            img_old = img
            mask_old = mask
            img = np.empty(img_old.shape)
            mask = np.empty(mask_old.shape)


            for iter_slice in range(self.num_slices):
                img_slice = img_old[:,:,iter_slice]
                mask_slice = mask_old[:,:,iter_slice]

                mask_slice = self.applyZoom( mask_slice , zf , 1 )
                img_slice = self.applyZoom( img_slice , zf , 0 )

                # self.print_range(img)

                mask_slice = self.applyRotation( mask_slice , theta , 1 )
                img_slice = self.applyRotation( img_slice , theta , 0 )

                # self.print_range(img)

                mask_slice = self.applyShift( mask_slice , tx, ty , 1 )
                img_slice = self.applyShift( img_slice , tx, ty , 0 )

                # self.print_range(img)


                if flipStackLR:
                    mask_slice = np.fliplr( mask_slice )
                    img_slice = np.fliplr( img_slice )

                if flipStackUD:
                    mask_slice = np.flipud( mask_slice )
                    img_slice = np.flipud( img_slice )

                img[:,:,iter_slice] = img_slice
                mask[:,:,iter_slice] =  mask_slice

        mask[mask>=0.5]=1.0
        mask[mask<0.5]=0.0


        img = img - np.amin(img)
        img = img / np.amax(img)

        return (img,mask)

    def print_range(self,img):

        print(np.amax(img))
        print(np.amin(img))



    def getRandomShift(self, h, w, widthShiftRange, heightShiftRange):
        # RANDOMLY DEFINES TRANSLATION IN X AND Y DIRECTIONS

        tx = self.random.uniform(-heightShiftRange, heightShiftRange) * h
        ty = self.random.uniform(-widthShiftRange, widthShiftRange) * w

        return (tx, ty)


    def getRandomFlipFlag(self):

        return self.random.choice([True, False])



    def getRandomZoomConfig(self, zoomRange ):
        if zoomRange[0] == 1 and zoomRange[1] == 1:
            zf = 1
        else:
            zf = self.random.uniform(zoomRange[0], zoomRange[1], 1)[0]
        return zf

    
    def getRandomRotation(self, rotationRange):
        # RANDOMLY GENERATES THE ROTATION THAT WILL BE USED

        theta = self.random.uniform(-rotationRange, rotationRange)
        return theta




    def applyZoom(self, img, zf, isMask , fill_mode='nearest', cval=0., interpolation_order=0):
        # BASED ON RANDOMLY DEFINED ZOOM VALUE FROM GETRANDOMZOOMCONFIG
        #
        #   APPLIES THE RANDOM ZOOM.
        #
        #   tHE SELF.ISMASK IS IMPORTANT PARAMETER BECAUSE IT DEFINES THE
        # INTERPOLATION TYPE. YOU ALWAYS WANT THE MASK TO REMAIN EITHER 0-1 OR 0-255 AND
        #       NOT BE ANY VALUES IN THE MIDDLE. HOWEVER, THE IMAGE CAN BE ANY UINT8
    

        if isMask:
            interp = cv2.INTER_NEAREST
            interpolation_order = 0
        else:
            interp = cv2.INTER_CUBIC
            interpolation_order = 1

        origShape = img.shape[1::-1]

        img = scipy.ndimage.zoom(img, zf, mode=fill_mode, cval=cval, order=interpolation_order)
        if zf < 1:
            canvas = np.zeros(origShape, dtype=img.dtype)
            rowOffset = int(np.floor((origShape[0] - img.shape[0])/2))
            colOffset = int(np.floor((origShape[1] - img.shape[1])/2))
            canvas[rowOffset:(rowOffset+img.shape[0]), colOffset:(colOffset+img.shape[1])] = img
            img = canvas
        elif zf > 1:
            rowOffset = int(np.floor((img.shape[0] - origShape[0])/2))
            colOffset = int(np.floor((img.shape[1] - origShape[1])/2))
            img = img[rowOffset:(rowOffset+origShape[0]), colOffset:(colOffset+origShape[1])]


        img = cv2.resize(img, origShape, interpolation=interp)

        return img



    def applyRotation(self, img, theta, isMask ):
        # APPLIES ROTATION ABOUT Z AXIS TO EACH IMAGE IN STACK

        """Performs a random rotation of a Numpy image tensor.
        # Arguments
            x: Input tensor. Must be 3D.
            rg: Rotation range, in degrees.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
            interpolation_order int: order of spline interpolation.
                see `ndimage.interpolation.affine_transform`
        # Returns
            Rotated Numpy image tensor.
        """
        

        if isMask:
            img_out = scipy.ndimage.rotate( img , theta , reshape=False , order=3 , 
                mode='constant' , cval=0 )

        else:
            img_out = scipy.ndimage.rotate( img , theta , reshape=False , 
                order=3 , mode='constant' , cval=0 )

        return img_out



    def applyShift(self, img, tx, ty, isMask , fill_mode='constant', cval=0., interpolation_order=0):
        # APPLIES TRANSLATION TO IMAGE SLIZE IN X AND Y DIRECTION

        """Performs a random spatial shift of a Numpy image tensor.
        # Arguments
            x: Input tensor. Must be 3D.
            wrg: Width shift range, as a float fraction of the width.
            hrg: Height shift range, as a float fraction of the height.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
            interpolation_order int: order of spline interpolation.
                see `ndimage.interpolation.affine_transform`
        # Returns
            Shifted Numpy image tensor.
        """
        img = scipy.ndimage.shift(img, [tx, ty], mode=fill_mode, cval=cval, order=interpolation_order)

        return img


    def display_2d_image_masks(self):

        print('\n\n SHOWING 2D IMAGES . . . \n\n')


        (imgs,masks)=self.get_batch()

        imgs = np.squeeze(imgs)
        masks = np.squeeze(masks)

        imgs = np.transpose(imgs,(1,2,0))
        masks = np.transpose(masks,(1,2,0))
        

        plot_matrix = np.hstack((imgs,masks))

        show_3d_images(plot_matrix)


    def display_3d_image_masks(self):

        print('\n\n SHOWING 3D IMAGES . . . \n\n')


        (imgs,masks)=self.get_batch()



        for counter in range(self.batch_size):
            img_tmp = np.squeeze(imgs[counter,:])
            mask_tmp = np.squeeze(masks[counter,:])
            
            plot_matrix = np.hstack((img_tmp,mask_tmp))

            show_3d_images(plot_matrix)
    def pass_info(self):

        return self.img_height, self.img_width,  self.train_scan_names, self.valid_scan_names




if __name__ == "__main__":
    

    # logger = get_logger('data_loader')
    # logger.info('Running data_loader')

    # parser = argparse.ArgumentParser(description='Please specify if you would like to use the center 2D slice or whole 3D volume for each scan')
    # parser.add_argument('--2d', dest='run_2d', action='store_true')
    # parser.add_argument('--3d', dest='run_2d', action='store_false')
    # parser.set_defaults(run_2d=True)

    # args = parser.parse_args()

    # run_2d = args.run_2d


    run_2d = False
    # run_2d = True

    zoom_range = (1,1)
    max_rotation_degree = 0
    width_shift_ratio = 0
    height_shift_ratio = 0
    flipud_bool = False
    fliplr_bool = False


    zoom_range = (0.9,1.1)
    max_rotation_degree = 15
    width_shift_ratio = 0.1
    height_shift_ratio = 0.1
    flipud_bool = True
    fliplr_bool = True

    the_generator = data_generator(
        bool_2d=run_2d,
        batch_size=5,
        bool_shuffle = True,
        zoomRange=zoom_range, 
        rotationRange=max_rotation_degree, 
        widthShiftRange=width_shift_ratio, 
        heightShiftRange=height_shift_ratio,
        flipLR=fliplr_bool, 
        flipUD=flipud_bool
        )

    if run_2d:
        the_generator.display_2d_image_masks()
    else:
        the_generator.display_3d_image_masks()

