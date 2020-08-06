#TODO: check if keras version matters
import tensorflow as tf
keras = tf.compat.v1.keras
Sequence = keras.utils.Sequence
import numpy as np 
import glob
import os
import keras
import gc
# import PIL
# from PIL import Image
# import Image
import scipy
import pydicom
import cv2
import re
#from tifffile import imread
# from skimage import skimage.transform 
import scipy.misc
import matplotlib.pyplot as plt

###############################################################################
###############################################################################
############### GEN DIRECTORY FUNCTIONS    ####################################
###############################################################################
###############################################################################


def getFileNamesFromDir(dirPath, namesOnly=True, fileExt=None):
	"""
	Note: make sure that the filenames are in a format such that the sorting function will return the desired order
	 (e.g. name using 0001, 0002, etc. instead of 1, 10, 11, 2, etc.
	"""
	if fileExt == None:
		filePathList = glob.glob(os.path.join(os.path.normpath(dirPath), "*"))
	else:
		filePathList = glob.glob(os.path.join(os.path.normpath(dirPath), "*"+fileExt))
	for filePath in filePathList:
		if not os.path.isfile(filePath):
			raise ValueError("%s is not a file" % filePath)
	if not namesOnly:
		return filePathList
	fileNameList = sorted([os.path.basename(os.path.normpath(path)) for path in filePathList])
	return fileNameList


###############################################################################
###############################################################################
###############     GENERATOR FUNCTIONS    ####################################
###############################################################################
###############################################################################

# def combine_generator( gen1 , gen2 ):
# 	# Attempting to combine the generators instead of zipping them
# 	while True:
# 		yield ( gen1.next() , gen2.next() )





class Generator_3D(Sequence):


	def __init__(self, scanList, dirPath, maskDir, numFramesPerStack, batchSize=32, dim=(224,224),
				 nChannels=1, seed=1, shuffle=False, sliceFileExt=".dcm", fitImgMethod="pad",
				 zoomRange=(1,1), rotationRange=0, widthShiftRange=0, heightShiftRange=0,
				 flipLR=False, flipUD=False, channel_order='first'):

		'Initialization'
		self.scanList = scanList
		self.dirPath = dirPath
		self.maskDir = maskDir
		self.dim = tuple(dim)
		self.batchSize = batchSize
		self.nChannels = nChannels
		self.shuffle = shuffle
		self.random = np.random.RandomState(seed=seed)
		self.on_epoch_end()
		self.numFramesPerStack = numFramesPerStack
		self.sliceFileExt = sliceFileExt
		self.fitImgMethod = fitImgMethod
		self.zoomRange = zoomRange
		self.rotationRange = rotationRange
		self.widthShiftRange = widthShiftRange
		self.heightShiftRange = -heightShiftRange
		self.flipLR = flipLR
		self.flipUD = flipUD
		self.channel_order = channel_order


	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.ceil(len(self.scanList) / self.batchSize))

	def __getitem__(self, index):
		print("\nCalling generator\n")

		scan_indices_to_grab = self.indexes[index*self.batchSize : (index+1)*self.batchSize]
		batchScanList = np.array(self.scanList)[scan_indices_to_grab]

		#If you run out of scans in scanList, fill the rest of the batch by re-using the early indices
		if len(batchScanList) < self.batchSize:
			extra_indices = self.indexes[:self.batchSize-len(batchScanList)]
			batchScanList = np.concatenate((batchScanList, np.array(self.scanList)[extra_indices]))

		Img , Mask = self.__data_generation(batchScanList)
		return Img , Mask


	def on_epoch_end(self):

		self.indexes = np.arange( len( self.scanList ) )
		if self.shuffle:
			self.random.shuffle(self.indexes)


	def __data_generation(self, batchScanList):
		# print('Calling data generation . . .')
		# print()

		#while True:
		#TODO: check that the while true is necessary
		# THIS WHILE TRUE IS CRUCIAL FOR THE KERAS TO RUN THE GENERATOR PROPERLY (?)

		'Generates data containing batchSize samples' # X : (n_samples, *dim, nChannels)
		# Initialization
		Img = np.empty((self.batchSize, self.nChannels, *self.dim, self.numFramesPerStack), dtype=np.float64)
		Mask = np.empty((self.batchSize, self.nChannels, *self.dim, self.numFramesPerStack), dtype=np.float64)

		# Generate data
		for counter in range(self.batchSize):
			scanName = batchScanList[counter]
			Img[counter,:,:,:,:] , Mask[counter,:,:,:,:] = self.getVolumeForFileID(scanName)
			#Note: this repeats the same volume across all of the channels
		if self.channel_order == "last":
			Img = np.moveaxis(Img, 1, -1)
			Mask = np.moveaxis(Mask, 1, -1)
		return Img , Mask



	def getVolumeForFileID(self, scanName):
		"""
		Generates volume
		"""
		Img_stack = np.empty((*self.dim, self.numFramesPerStack), dtype=np.float64)
		Mask_stack = np.empty((*self.dim, self.numFramesPerStack), dtype=np.float64)
		
		# Get the random augmentation states
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

		# Get each slice and apply augmentation
		sliceNames = getFileNamesFromDir(self.dirPath+scanName, fileExt=self.sliceFileExt)
		maskNames = getFileNamesFromDir(self.dirPath+scanName+'/'+self.maskDir)

		sliceStartIdx, sliceEndIdx = 0, self.numFramesPerStack-1
		if len(sliceNames) < self.numFramesPerStack:
			#Pad with empty slices to reach numFramesPerStack
			numExtra = self.numFramesPerStack - len(sliceNames)
			if numExtra == 1:
				sliceStartIdx = 1
			else:
				sliceStartIdx = int(np.ceil(numExtra/2))
				sliceEndIdx = sliceStartIdx + len(sliceNames) - 1
		for counter in range( self.numFramesPerStack ):
			if counter >= sliceStartIdx and counter <= sliceEndIdx:
				slicePath = self.dirPath+scanName+'/'+sliceNames[counter-sliceStartIdx]

				#Get original image
				if self.sliceFileExt == ".dcm":
					#Source images are DICOM images
					dataset = pydicom.dcmread(slicePath)
					Img_slice = dataset.pixel_array
				else:
					#Source images are non-DICOM images
					Img_slice = cv2.imread(slicePath)
					Img_slice = cv2.cvtColor(Img_slice , cv2.COLOR_BGR2GRAY)
					Img_slice = np.asarray(Img_slice)
				Img_slice = Img_slice.astype(np.float64)
				# To visualize:
				#import matplotlib.pyplot as plt
				#plt.imshow(Img_slice, cmap=plt.cm.bone)
				#plt.show(block=True)

				#Get mask image
				maskPath = self.dirPath+scanName+'/'+self.maskDir+'/'+maskNames[counter-sliceStartIdx]
				Mask_slice = cv2.imread(maskPath)
				Mask_slice = cv2.cvtColor(Mask_slice , cv2.COLOR_BGR2GRAY)
				Mask_slice = np.asarray(Mask_slice)

				#rescale Mask intensity to a 0-1 scale and apply a threshold so that all values >= 0.5 are set to 1
				Mask_slice = Mask_slice.astype(np.float64)
				if Mask_slice.flatten().max() > 0.:
					Mask_slice = Mask_slice / Mask_slice.flatten().max()
				Mask_slice[Mask_slice >= 0.5] = 1.

				if Img_slice.shape[0] < self.dim[0] or Img_slice.shape[1] < self.dim[1]:
					#Fit the image because its dimensions are less than self.dim
					if self.fitImgMethod == "interpolate":
						Img_slice = cv2.resize( Img_slice , self.dim , interpolation=cv2.INTER_CUBIC )
						Mask_slice = cv2.resize( Mask_slice , self.dim , interpolation=cv2.INTER_CUBIC )
					elif self.fitImgMethod == "pad":
						delta_h = self.dim[0] - Img_slice.shape[0]
						delta_w = self.dim[1] - Img_slice.shape[1]
						top, bottom = delta_h//2, delta_h-(delta_h//2)
						left, right = delta_w//2, delta_w-(delta_w//2)
						Img_slice = cv2.copyMakeBorder(Img_slice, top, bottom, left, right, cv2.BORDER_CONSTANT,
													   value=[0,0,0])
						Mask_slice = cv2.copyMakeBorder(Mask_slice, top, bottom, left, right, cv2.BORDER_CONSTANT,
														value=[0,0,0])
					else:
						raise Exception("fitImgMethod must be either 'interpolate' or 'pad'.")

				Mask_slice = self.applyZoom( Mask_slice , zf , 1 )
				Img_slice = self.applyZoom( Img_slice , zf , 0 )

				Mask_slice = self.applyRotation( Mask_slice , theta , 1 )
				Img_slice = self.applyRotation( Img_slice , theta , 0 )

				Mask_slice = self.applyShift( Mask_slice , tx, ty , 1 )
				Img_slice = self.applyShift( Img_slice , tx, ty , 0 )

				if flipStackLR:
					Img_slice = np.fliplr( Img_slice )
					Mask_slice = np.fliplr( Mask_slice )

				if flipStackUD:
					Img_slice = np.flipud( Img_slice )
					Mask_slice = np.flipud( Mask_slice )

			else:
				#Zero pad this entire slice
				Img_slice = np.zeros(self.dim)
				Mask_slice = np.zeros(self.dim)

			#Make sure that the mask array has values of either 0 or 1
			Mask_slice = Mask_slice.astype(np.float64)
			Mask_slice[Mask_slice < 0.5 ] = 0.
			Mask_slice[Mask_slice >= 0.5] = 1.

			Img_stack[ : , : , counter ] = Img_slice
			Mask_stack[ : , : , counter ] = Mask_slice

		# Normalization
		# TODO: consider different methods for MR normalization
		Img_slice = Img_slice.astype(np.float64)

		if Img_slice.flatten().min() < 0.:
			Img_slice += abs(Img_slice.flatten().min())
		else:
			Img_stack -= Img_stack.min()


		Img_stack = Img_stack / Img_stack.flatten().max()

		return  Img_stack , Mask_stack


	def getRandomFlipFlag(self):


		return self.random.choice([True, False])



	def getRandomZoomConfig(self, zoomRange ):
		if zoomRange[0] == 1 and zoomRange[1] == 1:
			zf = 1
		else:
			zf = self.random.uniform(zoomRange[0], zoomRange[1], 1)[0]
		return zf

	def applyZoom(self, img, zf, isMask , fill_mode='nearest', cval=0., interpolation_order=0):
		# BASED ON RANDOMLY DEFINED ZOOM VALUE FROM GETRANDOMZOOMCONFIG
		#
		#	APPLIES THE RANDOM ZOOM.
		#
		#	tHE SELF.ISMASK IS IMPORTANT PARAMETER BECAUSE IT DEFINES THE
		# INTERPOLATION TYPE. YOU ALWAYS WANT THE MASK TO REMAIN EITHER 0-1 OR 0-255 AND
		#		NOT BE ANY VALUES IN THE MIDDLE. HOWEVER, THE IMAGE CAN BE ANY UINT8
	

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


	
	def getRandomRotation(self, rotationRange):
		# RANDOMLY GENERATES THE ROTATION THAT WILL BE USED

		theta = self.random.uniform(-rotationRange, rotationRange)
		return theta



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


	def getRandomShift(self, h, w, widthShiftRange, heightShiftRange):
		# RANDOMLY DEFINES TRANSLATION IN X AND Y DIRECTIONS

		tx = self.random.uniform(-heightShiftRange, heightShiftRange) * h
		ty = self.random.uniform(-widthShiftRange, widthShiftRange) * w

		return (tx, ty)

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