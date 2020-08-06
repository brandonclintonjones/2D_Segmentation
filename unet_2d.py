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



		self.img_height = img_height
		self.img_width = img_width
		self.batch_size = batch_size
		self.learn_rate = learn_rate
		self.max_epoch = max_epoch
		self.model_name = model_name
		self.zoom_range = zoom_range
		self.max_rotation_degree = max_rotation_degree
		self.width_shift_ratio = width_shift_ratio
		self.height_shift_ratio = height_shift_ratio
		self.flipud_bool = flipud_bool
		self.fliplr_bool = fliplr_bool
		self.num_channels = num_channels



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

		(
		self.img_height,
		self.img_width,
		self.train_scan_names,
		self.valid_scan_names
		) = self.my_gen.pass_info()
		
		self.num_train_scans = len(self.train_scan_names)
		self.num_valid_scans = len(self.valid_scan_names)

		self.save_dir = os.getcwd()+'/Models/' + model_name +'/'

		print(self.save_dir)

		if not os.path.isdir(self.save_dir):
			os.makedirs(self.save_dir)


		self.dtype = tf.float32
		tf.logging.set_verbosity(tf.logging.ERROR)
		tf.set_random_seed(seed=1)


		config_options = tf.ConfigProto()
		# config_options = tf.ConfigProto(log_device_placement=True)
		# Log Device Placement prints which operations are performed

		# config_options.gpu_options.allow_growth = True
		config_options.gpu_options.allow_growth = False
		
		# Allow the GPU to use its maximum resources
		self.sess = tf.Session(config=config_options)
		# Define the Tensorflow Session



		self.input_matrix_shape = (
		    None,
		    self.num_channels,
		    self.img_height,
		    self.img_width,
		)


		self.input_image_placeholder = tf.placeholder(
		    dtype=self.dtype, shape=self.input_matrix_shape
		)

		self.mask_label_placeholder = tf.placeholder(
		    dtype=self.dtype, shape=self.input_matrix_shape
		)

		self.mask_predicted_placeholder = self.forward_pass(
		    tensor_input=self.input_image_placeholder
		)


		self.loss = self.custom_image_loss(
		    y_true=self.mask_label_placeholder,
		    y_pred=self.mask_predicted_placeholder,
		)


		self.optimizer_type = tf.train.AdamOptimizer(learning_rate=self.learn_rate)

		self.train_optimizer = self.optimizer_type.minimize(self.loss)

		init = tf.global_variables_initializer()

		self.sess.run(init)

		self.saver = tf.train.Saver()


	def forward_pass(self, tensor_input):
		"""
		    Runs the image through the layer structure.

		    """


		tensor_output = model_architectures.unet_9_layers(
		    tensor_input, output_tensor_channels=1
		)

		return tensor_output



	def train(self):



		save_str = self.save_dir + self.model_name

		steps_per_epoch_train = int(np.floor( self.num_train_scans / self.batch_size))
		steps_per_epoch_valid = int(np.floor( self.num_valid_scans / self.batch_size))
		

		epoch_loss_train = []
		epoch_loss_valid = []


		start_time = timeit.default_timer()

		for epoch_num in range(self.max_epoch):



			self.epoch_iter = epoch_num

			print("\n\n EPOCH NUMBER " + str(epoch_num + 1))



			# RUN TRAINING DATA THROUGH NETWORK AND BACKPROPAGATE GRADIENTS

			batch_loss_train = []
			batch_dsc_train = []


			for counter in tqdm.tqdm(range(steps_per_epoch_train)):


			    image_batch_train, mask_batch_train,  = self.my_gen.get_batch(
			        batch_ind=counter, is_train = True
			    )



			    tf_dict_train = {
			        self.input_image_placeholder: image_batch_train,
			        self.mask_label_placeholder: mask_batch_train,
			    }


			    # Run a forward pass and backpropagation and output the optimizer state and loss value
			    _ , loss_value_train = self.sess.run(
			        [self.train_optimizer, self.loss], tf_dict_train
			    )

			    mask_predicted_train = self.predict(mask_batch_train)

			    dsc_train = self.batch_dice_coef(y_true=mask_batch_train,y_pred=mask_predicted_train)

			    batch_loss_train.append(loss_value_train)
			    batch_dsc_train.append(dsc_train)



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


			    # Run the model on the validation set and report loss
			    # NOTE: DO NOT FEED OPTIMIZER IN THE VALIDATION TEST SO IT DOESNT UPDATE GRADIENTS

			    loss_value_valid = self.sess.run(
			        self.loss, tf_dict_valid
			    )

			    mask_predicted_valid = self.predict(mask_batch_valid)

			    if epoch_num % 4 == 0:
			    	p1 = image_batch_valid.squeeze().transpose((1,2,0))
			    	p2 = mask_batch_valid.squeeze().transpose((1,2,0))
			    	p3 = mask_predicted_valid.squeeze().transpose((1,2,0))
			    	p4 = p3 > 0.5


			    	tmp = np.hstack((p1, p2, p3, p4))
			    	# print(tmp.shape)
			    	show_3d_images(tmp)

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

		self.saver.save(self.sess, save_str, global_step=epoch_num + 1)



	def load(self):

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

        
	def load_submission(self, model_location):

		tf.reset_default_graph()

		meta_graph_name = model_location + "*.meta"

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

		if len(X_star.shape)==3:
		    bool_flip_output_dim = True
		    num_channels,img_height,img_width = X_star.shape
		    X_star = X_star.reshape((1,num_channels,img_height,img_width))

		predict_dictionary = {self.input_image_placeholder: X_star}

		predicted_mask = self.sess.run(
		    self.mask_predicted_placeholder, predict_dictionary
		)

		if bool_flip_output_dim:
		    predicted_mask = predicted_mask.reshape((num_channels,img_height,img_width))

		return predicted_mask


	def print_hist_info(self,mat):
		mat = mat.flatten()
		print(np.amax(mat))
		print(np.amin(mat))
		print(mat.mean())
		print('\n')


	def batch_dice_coef(self,y_true,y_pred):
		batch_dim,num_channels,img_height,img_width = y_true.shape

		# print('LABEL MAX MIN THEN PREDITION')
		# self.print_hist_info(y_true)
		# self.print_hist_info(y_pred)


		dsc_vals = []

		for img_iter in range(batch_dim):

			dsc_vals.append(self.dice_coef(y_true[img_iter,:],y_pred[img_iter,:]))

		dsc = np.asarray(dsc_vals).mean()
		return dsc



	def dice_coef(self,y_true,y_pred,smooth=1):
		# Expects numpy inputs
		
		y_pred[y_pred>=0.5] = 1.0
		y_pred[y_pred<0.5] = 0.0


		y_true = y_true.flatten()
		y_pred = y_pred.flatten()


		intersection = np.multiply(y_true,y_pred).sum()
		union = y_pred.sum() + y_true.sum()
		dsc = (2*intersection ) / (union )
		
		return dsc



	def custom_dice_loss(self,y_true,y_pred):

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
