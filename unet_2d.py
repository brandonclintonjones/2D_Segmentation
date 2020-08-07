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






class CNN():
	def __init__(self, 
		img_height,
		img_width,
		batch_size,
		learn_rate,
		max_epoch,
		model_name,
		num_channels = 1, # could be 3 or something if RGB
		zoom_range = (1,1),
		max_rotation_degree = 0,
		width_shift_ratio = 0,
		height_shift_ratio = 0,
		flipud_bool = False,
		fliplr_bool = False,
		 ):


		"""
		
		NOTE: This is the CNN Class Constructor
		Takes inputs about image dimensions, network parameters (batch size, learn rate, max epochs), and 
		data augmentation parameters (booleans for flip up/down and left/right )

		All input parameters are then stored as objects of the class for convenient calling later by different
		class methods


		"""


		### ///////////
		###
		###   Input parameters pertaining to the images
		###
		### /////////// 


		self.img_height = img_height
		self.img_width = img_width
		self.num_channels = num_channels


		### ///////////
		###
		###   Input parameters pertaining to the network training
		###
		### /////////// 



		self.batch_size = batch_size
			# Usually 1-2 for 3D, up to 10 for 2D

		self.learn_rate = learn_rate
			# Typically 1e-3 to 1e-5

		self.max_epoch = max_epoch
			# Like 50ish for 2D

		self.model_name = model_name 
			# Example: b_8_e_50_lr_1e-05
			# Denoting batch = 8, epoch 50, and so on

		
		### ///////////
		###
		###   Input parameters pertaining to the data augmentation
		###   These are directly put into the class for data augmentation in the file 
		###   			custom_generators.py   
		####
		### /////////// 


		self.zoom_range = zoom_range
			# Tuple like (1,1) or (0.9,1.1) indicating no zoom or 90-110%
		self.max_rotation_degree = max_rotation_degree
			# Degree of max rotation. 0 or 30 or 90
		self.width_shift_ratio = width_shift_ratio
			# Fraction shift for augmentation. 0.1 means 10% of matrix pixels
		self.height_shift_ratio = height_shift_ratio
		self.flipud_bool = flipud_bool
			# Self explanatory. Do you want to flip up-down or left-right?
		self.fliplr_bool = fliplr_bool

		# Instantiate the data generator class

		self.my_gen = data_generator(
			bool_2d=True,
			batch_size=batch_size,
			bool_shuffle = True,
			zoomRange=zoom_range, 
			rotationRange=max_rotation_degree, 
			widthShiftRange=width_shift_ratio, 
			heightShiftRange=height_shift_ratio,
			flipLR=fliplr_bool, 
			flipUD=flipud_bool
			)

		# Pass some info between the two

		(
		self.img_height,
		self.img_width,
		self.train_scan_names,
		self.valid_scan_names
		) = self.my_gen.pass_info()
		


		# Get the number of training and validation scans and save as class object
		self.num_train_scans = len(self.train_scan_names)
		self.num_valid_scans = len(self.valid_scan_names)

		# Create save file string for where the model and weight checkpoints will be saved
		# If the directory doesnt exist, create it

		self.save_dir = os.getcwd()+'/Models/' + model_name +'/'

		print(self.save_dir)

		if not os.path.isdir(self.save_dir):
			os.makedirs(self.save_dir)


		# Tensorflow default data type. Useful later on for casting between NumPy objects and TensorFlow objects
		self.dtype = tf.float32

		# Random TensorFlow initial settings
		tf.logging.set_verbosity(tf.logging.ERROR)
		tf.set_random_seed(seed=1)


		config_options = tf.ConfigProto()
		config_options.gpu_options.allow_growth = True
		os.environ["CUDA_VISIBLE_DEVICES"]="0"
		# Allow the GPU to use its maximum resources

		# This is the main Tensorflow call. "SESSION" is their term for
		# the main TF caller

		self.sess = tf.Session(config=config_options)

		
		### ///////////
		###
		###   annnddddd now we initialize the actual network graph
		###   Define tensor dimensions and important operations   
		###
		### /////////// 


		# NOTE: THE WAY I'VE CODED THIS, THE INPUTS AND OUTPUTS ARE ALWAYS NUMPY OBJECTS
		# WHICH ARE SUPER EASY TO WORK WITH. THE CLASS AUTOMATICALLY CASTS TO TENSORS FOR 
		# OPERATIONS WHERE THATS NEEDED. 


		self.input_matrix_shape = (
		    None,
		    self.num_channels,
		    self.img_height,
		    self.img_width,
		)
		# ( None, 1, 256, 256 )
		# ( Batch , channels, height, width )


		#
		# NOW DEFINE THE EMPTY TENSOR PLACEHOLDERS FOR THE NETWORK GRAPH
		#
		#  NEED PLACEHOLDERS FOR THE INPUT IMAGE, THE GROUND TRUTH MASK, AND THE 
		#	PREDICTED MASK
		#

		self.input_image_placeholder = tf.placeholder(
		    dtype=self.dtype, shape=self.input_matrix_shape
		)

		self.mask_label_placeholder = tf.placeholder(
		    dtype=self.dtype, shape=self.input_matrix_shape
		)

		# NOTICE THE DIFFERENCE HERE - THE PLACEHOLDER FOR PREDICTION IS FORWARD PASS OF THE MODEL
		self.mask_predicted_placeholder = self.forward_pass(
		    tensor_input=self.input_image_placeholder
		)

		# MY CUSTOM IMAGE LOSS. SEE THE METHOD AT THE BOTTOM. THIS PART IS COMPLICATED
		self.loss = self.custom_image_loss(
		    y_true=self.mask_label_placeholder,
		    y_pred=self.mask_predicted_placeholder,
		)


		# Define the optimizer. By default this is just Adam in most cases
		self.optimizer_type = tf.train.AdamOptimizer(learning_rate=self.learn_rate)

		# Tell the optimizer what objective function is should optimize
		self.train_optimizer = self.optimizer_type.minimize(self.loss)

		# Initialize variables. The method of initialization is defined in the "Model_architectures.py"
		# So like in a conv instantiation we will tell it "Gorot" or "Gaussian" or whatever
		init = tf.global_variables_initializer()

		self.sess.run(init)

		self.saver = tf.train.Saver()


	def forward_pass(self, tensor_input):
		"""
		    Runs the image through the layer structure.
			Due to the complexity of the network graph, this is abstracted to another python file
			specifically dedicated to network graphs

		    """


		tensor_output = model_architectures.unet_9_layers(
		    tensor_input, output_tensor_channels=1
		)

		return tensor_output



	def train(self):
		"""
		    OK SO THIS IS THE MAIN MEAT OF THE CODE. IN GENERAL, ITS A GIANT FOR LOOP OVER THE NUMBER OF EPOCHS

			AT EVERY ITERATION, A TRAINING BATCH IS GENERATED,
			THE IMAGES ARE FORWARD PASSED, AND THE GRADIENT IS BACKPROPAGATED
			
			THEN THE VALIDATION DATA IS LOADED AND THE ACCURACY AND LOSS ARE RECORDED FOR 
			THE USER'S SAKE

			

		    """


		save_str = self.save_dir + self.model_name

		steps_per_epoch_train = int(np.floor( self.num_train_scans / self.batch_size))
		steps_per_epoch_valid = int(np.floor( self.num_valid_scans / self.batch_size))
		
		# Define empty lists for the losses for convenient recording
		epoch_loss_train = []
		epoch_loss_valid = []

		# Tracks timee for each iteration
		start_time = timeit.default_timer()


		# NOTE: THIS IS THE MAIN TRAINING LOOP
		for epoch_num in range(self.max_epoch):



			self.epoch_iter = epoch_num

			print("\n\n EPOCH NUMBER " + str(epoch_num + 1))



			# RUN TRAINING DATA THROUGH NETWORK AND BACKPROPAGATE GRADIENTS

			batch_loss_train = []
			batch_dsc_train = []

			# This iterates through all batches of training data for one epoch
			# TQDM is just used to display a pretty loading bar while training

			for counter in tqdm.tqdm(range(steps_per_epoch_train)):


				# Load the batch from the generator
			    image_batch_train, mask_batch_train,  = self.my_gen.get_batch(
			        batch_ind=counter, is_train = True
			    )


				# Create a dictionary of the Numpy arrays to feed into the tensor
			    tf_dict_train = {
			        self.input_image_placeholder: image_batch_train,
			        self.mask_label_placeholder: mask_batch_train,
			    }


			    # Run a forward pass and backpropagation and output the optimizer state and loss value
				# NOTE: THIS FEEDS IN THE OPTIMIZER AND THE LOSS FUNCTION, AS WELL AS 
				# THE DATA DICTIONARY
			    _ , loss_value_train = self.sess.run(
			        [self.train_optimizer, self.loss], tf_dict_train
			    )

				# This just applies a second forward pass so I can report the Dice similarity score,
				# which is a much more informative measure of the network accuracy than the "loss" which is 
				# arbitrarily unscaled
			    mask_predicted_train = self.predict(mask_batch_train)

				# Compute DSC
			    dsc_train = self.batch_dice_coef(y_true=mask_batch_train,y_pred=mask_predicted_train)

				# Add the loss and DSC (accuracy) to the lists and move through loop again
			    batch_loss_train.append(loss_value_train)
			    batch_dsc_train.append(dsc_train)


			# Convert the lists to numpy arrays and compute the mean over the epoch
			batch_loss_avg_train = np.asarray(batch_loss_train).mean()
			batch_dsc_avg_train = np.asarray(batch_dsc_train).mean()

			elapsed = timeit.default_timer() - start_time

			print(
			    "TRAIN ==> Epoch [%d/%d], Loss: %.12f, DSC: %.12f, Time: %2fs"
			    % (epoch_num + 1, self.max_epoch, batch_loss_avg_train, batch_dsc_avg_train, elapsed)
			)

			start_time = timeit.default_timer()

			epoch_loss_train.append(batch_loss_avg_train)



			# RUN VALIDATION SET THROUGH MODEL AND REPORT ACCURACY AND LOSS
			batch_loss_valid = []
			batch_dsc_valid = []


			for counter in tqdm.tqdm(range(steps_per_epoch_valid)):


			    image_batch_valid, mask_batch_valid,  = self.my_gen.get_batch(
			        batch_ind=counter, is_train = False
			    )


			    tf_dict_valid = {
			        self.input_image_placeholder: image_batch_valid,
			        self.mask_label_placeholder: mask_batch_valid,
			    }


			    
			    # NOTE: SAME AS BEFORE, BUT WE DO NOT FEED IN THE OPTIMIZER, 
				# 		SINCE THE WHOLE POINT OF VALIDATION SET IS ITS NOT TRAINED ON VALID DATA
			    loss_value_valid = self.sess.run(
			        self.loss, tf_dict_valid
			    )

			    mask_predicted_valid = self.predict(mask_batch_valid)

				# Ignore this, this is me plotting to debug training
				# ////
			    if epoch_num % 4 == 0:
			    	p1 = image_batch_valid.squeeze().transpose((1,2,0))
			    	p2 = mask_batch_valid.squeeze().transpose((1,2,0))
			    	p3 = mask_predicted_valid.squeeze().transpose((1,2,0))
			    	p4 = p3 > 0.5


			    	tmp = np.hstack((p1, p2, p3, p4))
			    	# print(tmp.shape)
			    	show_3d_images(tmp)

				# /////

			    dsc_valid = self.batch_dice_coef(y_true=mask_batch_valid,y_pred=mask_predicted_valid)

			    batch_loss_valid.append(loss_value_valid)
			    batch_dsc_valid.append(dsc_valid)



			batch_loss_avg_valid = np.asarray(batch_loss_valid).mean()
			batch_dsc_avg_valid = np.asarray(batch_dsc_valid).mean()

			elapsed = timeit.default_timer() - start_time

			print(
			    "VALID ==> Epoch [%d/%d], Loss: %.12f, DSC: %.12f, Time: %2fs"
			    % (epoch_num + 1, self.max_epoch, batch_loss_avg_valid, batch_dsc_avg_valid, elapsed)
			)

			start_time = timeit.default_timer()

			epoch_loss_valid.append(batch_loss_avg_valid)




			if (epoch_num + 1) % 10 == 0:
			    print("\n\n SAVING MODEL . . . \n\n")

			    self.saver.save(self.sess, save_str, global_step=epoch_num + 1)


		# Save the model to the desired directory
		self.saver.save(self.sess, save_str, global_step=epoch_num + 1)



	def load(self):
		'''
		
		IF WE WANT TO LOAD THE MODEL TO CONTINUE TRAINING OR PREDICT WE CALL THIS METHOD

		'''

		tf.reset_default_graph()

		meta_graph_name = self.save_dir + self.model_name + "*.meta"

		files_in_dir = glob.glob(meta_graph_name)

		num_files = len(files_in_dir)

		meta_graph_name = files_in_dir[num_files - 1]
		# Grabs the last saved checkpoint ih the directory. Assuming last one is
		# the most trained one

		self.save_dir = os.path.dirname(meta_graph_name)

		self.save_model_name = meta_graph_name[0 : len(meta_graph_name) - 5]

		self.saver = tf.train.import_meta_graph(meta_graph_name)

		self.saver.restore(self.sess, self.save_model_name)


	def predict(self, X_star):

		bool_flip_output_dim = False

		# If we feed in a single image, we slightly modify the image dimensions before applying the operations

		if len(X_star.shape)==3:
		    bool_flip_output_dim = True
		    num_channels,img_height,img_width = X_star.shape
		    X_star = X_star.reshape((1,num_channels,img_height,img_width))


		# Define a dictionary of just the input image "X_star"
		# Unlike in training phase, we don't feed in the 
		predict_dictionary = {self.input_image_placeholder: X_star}

		predicted_mask = self.sess.run(
		    self.mask_predicted_placeholder, predict_dictionary
		)


		# IF we changed the dimensions, we reshape back here
		if bool_flip_output_dim:
		    predicted_mask = predicted_mask.reshape((num_channels,img_height,img_width))


		return predicted_mask


	def print_hist_info(self,mat):
		# Ignore, useful for debugging
		mat = mat.flatten()
		print(np.amax(mat))
		print(np.amin(mat))
		print(mat.mean())
		print('\n')


	def batch_dice_coef(self,y_true,y_pred):
		# Compute the dice similarity score for the batch
		batch_dim,num_channels,img_height,img_width = y_true.shape

		dsc_vals = []

		for img_iter in range(batch_dim):

			dsc_vals.append(self.dice_coef(y_true[img_iter,:],y_pred[img_iter,:]))

		dsc = np.asarray(dsc_vals).mean()
		return dsc



	def dice_coef(self,y_true,y_pred,smooth=1):
		# Compute DSC
		
		y_pred[y_pred>=0.5] = 1.0
		y_pred[y_pred<0.5] = 0.0


		y_true = y_true.flatten()
		y_pred = y_pred.flatten()


		intersection = np.multiply(y_true,y_pred).sum()
		union = y_pred.sum() + y_true.sum()
		dsc = (2*intersection ) / (union )
		
		return dsc



	def custom_dice_loss(self,y_true,y_pred):

		# NOTE: DEPRACATED, NOT USING THIS ANYMORE
		smooth = 1

		# NOTE: SUPER DUMB WAY TO TRUNCATE THE PREDICTED Y MATRIX TO 0 AND 1 
		# WHILE STILL BEING DIFFERENTIABLE IN TENSORFLOW
		# AT LEAST SYMBOLICALLY, DSC IS NOT TECHNICALLY DIFFERENTIABLE BECAUSE OF THE FLOOR FUNCTION
		y_pred_bool = tf.maximum(y_pred-0.499,0)
		y_pred_bool = tf.minimum(y_pred_bool*10000,1)

		intersection = tf.reduce_sum(tf.multiply(y_pred_bool,y_true))
		union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bool)
		numerator = 2 * intersection + smooth
		denominator = union + smooth
		dsc = tf.divide(numerator,denominator)

		return 1 - tf.divide(numerator,denominator)
		# return intersection




	def cross_entropy_weighted(self,y_true,y_pred):

		'''

			THIS IS SUPER IMPORTANT BUT IS COMPLICATED SO TALK TO BRANDON IN PERSON IF NEEDED

		'''
		smooth = 1

		jitter = 1e-8

		num_bone = tf.reduce_sum(y_true)
		num_background = tf.reduce_sum(1 - y_true)

		total_num = num_bone + num_background

		# first_term = tf.reduce_sum( y_true * tf.log(y_pred + jitter ) )
		# second_term =  tf.reduce_sum( (1 - y_true) * tf.log( 1 - y_pred + jitter ) )

		first_term = 10  * tf.reduce_sum( y_true * tf.log(y_pred + jitter ) )
		second_term = tf.reduce_sum( (1 - y_true) * tf.log( 1 - y_pred + jitter ) )

		# first_term = num_background * tf.reduce_sum( y_true * tf.log(y_pred + jitter ) )
		# second_term = num_bone * tf.reduce_sum( (1 - y_true) * tf.log( 1 - y_pred + jitter ) )


		weighted_cross_entropy = -1 * ( first_term + second_term + jitter ) / total_num

		return weighted_cross_entropy

	def mean_absolute_error(self,y_true,y_pred):
		# Mean L1 error

		total_1 = tf.reduce_sum(y_true)
		total_0 = tf.reduce_sum(y_true)
		total = total_0 + total_1

		mae = tf.divide(tf.reduce_sum(tf.math.abs(y_true-y_pred)),
			total)

		return mae


	def custom_image_loss(self, y_true, y_pred):

		# dice_loss = self.custom_dice_loss(y_true,y_pred)

		# bce = tf.keras.losses.binary_crossentropy(
		# 	self.mask_label_placeholder,
		# 	self.mask_predicted_placeholder
		# 	)

		# mse = tf.losses.mean_pairwise_squared_error(
		# 	labels = y_true,
		# 	predictions = y_pred 
		# 	)

		# mse = tf.losses.mean_squared_error(
		# 	labels = y_true,
		# 	predictions = y_pred
		# 	)

		# bce = tf.losses.sigmoid_cross_entropy(
		# 	multi_class_labels = y_true,
		# 	logits = y_pred )

		# hinge = tf.losses.hinge_loss(
		# 	labels = y_true,
		# 	logits = y_pred
		# 	)

		mae = self.mean_absolute_error(
			y_true = y_true,
			y_pred = y_pred
			)

		ce_weighted = self.cross_entropy_weighted(
			y_true=y_true,
			y_pred=y_pred
				)

		loss = ce_weighted + mae
		# loss = hinge
		# loss = mae

		return loss


def main():
	"""
	    Tests the CNN.

	    """

	# CNN Parameters


	run_2d = True    
	batch_size = 8
	max_epoch = 50
	lr = 1e-5

	# GENERATOR PARAMETERS
	# zoom_range = (0.95,1.05)
	# max_rotation_degree = 10
	# width_shift_ratio = 0.05
	# height_shift_ratio = 0.05
	# flipud_bool = True
	# fliplr_bool = True

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
	train_scan_names,
	valid_scan_names
	) = the_generator.pass_info()



	# This creates the model name based on the network training parameters
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

	# convnet.load()

	convnet.train()





if __name__ == "__main__":
    
    main()
