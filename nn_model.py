from __future__ import absolute_import, division, print_function

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Reshape, LSTM, GRU, Dropout, Add, multiply, add, dot, TimeDistributed, subtract
from keras.layers import Conv2D, BatchNormalization, ReLU, LeakyReLU, Conv2DTranspose, concatenate, Concatenate, Bidirectional, MaxPooling2D
from keras.layers import Activation, ConvLSTM2D, Permute
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras import losses
from keras.utils import plot_model
from keras.regularizers import l1, l2
from keras.utils.training_utils import multi_gpu_model

import numpy as np

import matplotlib.pyplot as plt

print(tf.__version__)

def build_model_dae(input_shape, model_type, tensor_B_x, tensor_B_d, BATCH_SIZE, timestep, ngpu):

	if model_type == 'U-Net':

		# Parameters
		b, k, r_x = tensor_B_x.shape
		b, k, r_d = tensor_B_d.shape

		#----U-net Encoder----
		y = Input(shape=(timestep,512,1))#shape=input_shape ) #(B,t=128,f=512, c=1)
		conv1 = conv_bat_relu(y,16, (5,5), 2) #(1,64,256,16)
		conv2 = conv_bat_relu(conv1,32, (5,5), 2) #(1,32,128,32)
		conv3 = conv_bat_relu(conv2,64, (5,5), 2) #(1,16,64,64)
		conv4 = conv_bat_relu(conv3,128, (5,5), 2) #(1,8,32,128)
		conv5 = conv_bat_relu(conv4,256, (5,5), 2) #(1,4,16,256)
		conv6 = conv_bat_relu(conv5,512, (5,5), 2) #(1,2,8,512)
		
		Unet_encoder = Model(y, [conv6, conv5, conv4, conv3, conv2, conv1], name = 'Enc_codes')  
		Unet_encoder.summary()
		#plot_model(Unet_encoder, to_file='model/Unet_encoder.png', show_shapes=True)

		#----U-net Decoder----
		d_y = Input(shape=( timestep,512,1)) #(B,t=8,f=512, c=1)
		d_temporal_codes = Input(shape=( timestep, r_x+r_d)) #(B, timestep * r_x)
		
		d_conv6 = Input(shape=( 2,8,512))
		d_conv5 = Input(shape=( 4,16,256))
		d_conv4 = Input(shape=( 8,32,128)) 
		d_conv3 = Input(shape=( 16,64,64))
		d_conv2 = Input(shape=( 32,128,32))
		d_conv1 = Input(shape=( 64,256,16))

		conc0 = d_conv6
		deconv1 = deconv_bat_relu(conc0,256, (5,5), 2)
		conc1 = concatenate([deconv1,d_conv5])
		deconv2 = deconv_bat_relu(conc1, 128, (5,5), 2)
		conc2 = concatenate([deconv2,d_conv4])
		deconv3 = deconv_bat_relu(conc2, 64, (5,5), 2)
		conc3 = concatenate([deconv3,d_conv3])
		deconv4 = deconv_bat_relu(conc3, 32, (5,5), 2)
		conc4 = concatenate([deconv4,d_conv2])
		deconv5 = deconv_bat_relu(conc4, 16, (5,5), 2)
		conc5 = concatenate([deconv5, d_conv1])
		irm_x = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation='sigmoid')(conc5)
		irm_d = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation='sigmoid')(conc5)
		x_hat = multiply([irm_x, d_y])
		d_hat = multiply([irm_d, d_y])

		#U-net Decoder
		Unet_decoder = Model([d_y, \
								d_conv6, d_conv5, d_conv4, d_conv3, d_conv2, d_conv1], [x_hat, d_hat], name = 'Dec_outs')
		Unet_decoder.summary()
		#plot_model(TAUnet_decoder, to_file='model/Unet_decoder.png', show_shapes=True)

		#U-net Model
		B_x = K.constant(tensor_B_x)
		B_d = K.constant(tensor_B_d)
		B_x = Input(tensor=B_x) #(B, 512, 100)
		B_d = Input(tensor=B_d) #(B, 512, 100)

		enc_outs = Unet_encoder(y)
		dec_outs = Unet_decoder([y, *enc_outs])
		x_hat_m = Reshape((timestep,k,1), name='Out_x_hat')(dec_outs[0])
		d_hat_m = Reshape((timestep,k,1), name='Out_d_hat')(dec_outs[1])
		
		Unet = Model([y, B_x, B_d], [x_hat_m, d_hat_m], name='Unet')
		Unet.summary()
		#plot_model(Unet, to_file='model/discrim_sep_TAUnet.png', show_shapes=True)

		# Define loss
		def stft_losses_dec_x(x, _):
			rmse_x = losses.mean_squared_error(x, x_hat_m)
			mae_x = losses.mean_absolute_error(x, x_hat_m)
			loss_x = rmse_x + mae_x
			rmse_neg_x = losses.mean_squared_error(x, d_hat_m)
			mae_neg_x = losses.mean_absolute_error(x, d_hat_m)
			loss_neg_x = rmse_neg_x + mae_neg_x

			total_loss = loss_x# - 0.01*loss_neg_x
			return total_loss

		# Define loss
		def stft_losses_dec_d(d, _):
			rmse_d = losses.mean_squared_error(d, d_hat_m)
			mae_d = losses.mean_absolute_error(d, d_hat_m)
			loss_d = rmse_d + mae_d
			rmse_neg_d = losses.mean_squared_error(d, x_hat_m)
			mae_neg_d = losses.mean_absolute_error(d, x_hat_m)
			loss_neg_d = rmse_neg_d + mae_neg_d
		
			total_loss = loss_d# - 0.01*loss_neg_d
			return total_loss

		def stft_mag_metric(x, x_hat):
			metric = losses.mean_absolute_error(x, x_hat)

			return metric / BATCH_SIZE    

		Unet.compile( loss= {'Out_x_hat' : stft_losses_dec_x,
							   'Out_d_hat' : stft_losses_dec_d},
						loss_weights = {'Out_x_hat' : 1.0,
										'Out_d_hat' : 1.0},
						optimizer='adam',
						#optimizer=optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
						metrics=[stft_mag_metric] )
		Unet.summary()
		model = Unet
		encoder = Unet_encoder
		decoder = Unet_decoder

	#------------TAU-Net networks------------
	if model_type == 'TAU-Net':

		# Parameters
		b, k, r_x = tensor_B_x.shape
		b, k, r_d = tensor_B_d.shape
		latent_dim_x = timestep * r_x #patch frame number * dim
		latent_dim_d = timestep * r_d #patch frame number * dim

		#----TAU-net Encoder----
		y = Input(shape=(timestep,512,1))#shape=input_shape ) #(B,t=128,f=512, c=1)
		conv1 = conv_bat_relu(y,16, (5,5), 2) #(1,64,256,16)
		conv2 = conv_bat_relu(conv1,32, (5,5), 2) #(1,32,128,32)
		conv3 = conv_bat_relu(conv2,64, (5,5), 2) #(1,16,64,64)
		conv4 = conv_bat_relu(conv3,128, (5,5), 2) #(1,8,32,128)
		conv5 = conv_bat_relu(conv4,256, (5,5), 2) #(1,4,16,256)
		conv6 = conv_bat_relu(conv5,512, (5,5), 2) #(1,2,8,512)
		codes = Reshape((timestep,64))(conv6) #(B,T=128,64)
		
		# Code Layer
		temporal_code = Bidirectional(LSTM(256, activation='relu', return_sequences=True, stateful=False, 
				kernel_regularizer=l2(1e-5), activity_regularizer=l2(1e-5)), merge_mode='ave')(codes) #(B,T,256)
		temporal_code = Bidirectional(LSTM(128, activation='relu', return_sequences=True, stateful=False, 
				kernel_regularizer=l2(1e-5), activity_regularizer=l2(1e-5)), merge_mode='ave')(temporal_code) #(B,T,128)
		temporal_code = Dropout(rate=0.01)(temporal_code)

		TAUnet_encoder = Model(y, [temporal_code, conv6, conv5, conv4, conv3, conv2, conv1], name = 'Enc_codes')  
		TAUnet_encoder.summary()
		#plot_model(TAUnet_encoder, to_file='model/TAUnet_encoder.png', show_shapes=True)


		#----TAU-net Decoder----
		d_y = Input(shape=( timestep,512,1)) #(B,t=8,f=512, c=1)
		d_temporal_codes = Input(shape=( timestep, r_x+r_d)) #(B, timestep * r_x)
		
		d_conv6 = Input(shape=( 2,8,512))
		d_conv5 = Input(shape=( 4,16,256))
		d_conv4 = Input(shape=( 8,32,128)) 
		d_conv3 = Input(shape=( 16,64,64))
		d_conv2 = Input(shape=( 32,128,32))
		d_conv1 = Input(shape=( 64,256,16))

		d_codes = Bidirectional(LSTM(256, activation='relu', return_sequences=True, stateful=False, 
				kernel_regularizer=l2(1e-5), activity_regularizer=l2(1e-5)), merge_mode='ave')(d_temporal_codes) #(B,T,256)
		d_codes = Bidirectional(LSTM(64, activation='relu', return_sequences=True, stateful=False, 
				kernel_regularizer=l2(1e-5), activity_regularizer=l2(1e-5)), merge_mode='ave')(d_codes) #(B,T,256)
		d_codes = Reshape((2,8,512))(d_codes)
		
		#Code chainer
		#d_codes2 = deconv_bat_relu(d_codes,256, (5,5), 2)
		#d_codes3 = deconv_bat_relu(d_codes2, 128, (5,5), 2)
		#d_codes4 = deconv_bat_relu(d_codes3, 64, (5,5), 2)
		#d_codes5 = deconv_bat_relu(d_codes4, 32, (5,5), 2)
		#d_codes6 = deconv_bat_relu(d_codes5, 16, (5,5), 2)

		#d_conv6_m = multiply([d_conv6,d_codes])
		#d_conv5_m = multiply([d_conv5,d_codes2])
		#d_conv4_m = multiply([d_conv4,d_codes3])
		#d_conv3_m = multiply([d_conv3,d_codes4])
		#d_conv2_m = multiply([d_conv2,d_codes5])
		#d_conv1_m = multiply([d_conv1,d_codes6])
		d_conv6_m = d_conv6
		d_conv5_m = d_conv5
		d_conv4_m = d_conv4
		d_conv3_m = d_conv3
		d_conv2_m = d_conv2
		d_conv1_m = d_conv1


		#conc0 = concatenate([d_codes,d_conv6_m])
		conc0 = d_codes
		deconv1 = deconv_bat_relu(conc0,256, (5,5), 2)
		conc1 = concatenate([deconv1,d_conv5_m])
		deconv2 = deconv_bat_relu(conc1, 128, (5,5), 2)
		conc2 = concatenate([deconv2,d_conv4_m])
		deconv3 = deconv_bat_relu(conc2, 64, (5,5), 2)
		conc3 = concatenate([deconv3,d_conv3_m])
		deconv4 = deconv_bat_relu(conc3, 32, (5,5), 2)
		conc4 = concatenate([deconv4,d_conv2_m])
		deconv5 = deconv_bat_relu(conc4, 16, (5,5), 2)
		conc5 = concatenate([deconv5, d_conv1_m])
		irm_x = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation='sigmoid')(conc5)
		irm_d = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation='sigmoid')(conc5)
		x_hat = multiply([irm_x, d_y])
		d_hat = multiply([irm_d, d_y])

		#TAU-net Decoder
		TAUnet_decoder = Model([d_y, d_temporal_codes, \
								d_conv6, d_conv5, d_conv4, d_conv3, d_conv2, d_conv1], [x_hat, d_hat], name = 'Dec_outs')
		TAUnet_decoder.summary()
		#plot_model(TAUnet_decoder, to_file='model/TAUnet_decoder.png', show_shapes=True)

		#TAU-net Model
		B_x = K.constant(tensor_B_x)
		B_x = K.repeat_elements(B_x,rep=BATCH_SIZE,axis=0)
		B_d = K.constant(tensor_B_d)
		B_d = K.repeat_elements(B_d,rep=BATCH_SIZE,axis=0)
		B_x = Input(tensor=B_x) #(B, 512, 100)
		B_d = Input(tensor=B_d) #(B, 512, 100)

		enc_outs = TAUnet_encoder(y)
		dec_outs = TAUnet_decoder([y, *enc_outs])
		x_hat_m = Reshape((timestep,k,1), name='Out_x_hat')(dec_outs[0])
		d_hat_m = Reshape((timestep,k,1), name='Out_d_hat')(dec_outs[1])
		A_tot = Reshape((timestep,r_x+r_d))(enc_outs[0]) #(B,T,100)
		A_x = Lambda(lambda x:x[:,:,:r_x] + 1e-6)(A_tot)
		A_d = Lambda(lambda x:x[:,:,r_x:] + 1e-6)(A_tot)
		#x_NMF = Lambda(lambda x:K.batch_dot(x[0],x[1], axes=2))([B_x, A_x])
		#d_NMF = Lambda(lambda x:K.batch_dot(x[0],x[1], axes=2))([B_d, A_d])
		#x_NMF = Permute((2,1))(x_NMF)
		#d_NMF = Permute((2,1))(d_NMF)

		x_NMF = dot([A_x, B_x], axes=2) #(B,T,512)
		d_NMF = dot([A_d, B_d], axes=2) #(B,T,512)
		x_NMF = Reshape((timestep,k,1), name = 'Out_x_NMF')(x_NMF)
		d_NMF = Reshape((timestep,k,1), name = 'Out_d_NMF')(d_NMF)
		
		TAUnet = Model([y, B_x, B_d], [x_hat_m, d_hat_m, x_NMF, d_NMF], name='TAUnet')
		TAUnet.summary()
		#plot_model(TAUnet, to_file='model/discrim_sep_TAUnet.png', show_shapes=True)

		# Define loss
		def stft_losses_dec_x(x, _):
			rmse_x = losses.mean_squared_error(x, x_hat_m)
			mae_x = losses.mean_absolute_error(x, x_hat_m)
			loss_x = mae_x + rmse_x
			rmse_neg_x = losses.mean_squared_error(x, d_hat_m)
			mae_neg_x = losses.mean_absolute_error(x, d_hat_m)
			loss_neg_x = mae_neg_x + rmse_neg_x

			total_loss = K.abs(loss_x - 0.01*loss_neg_x)
			return total_loss

		# Define loss
		def stft_losses_dec_d(d, _):
			rmse_d = losses.mean_squared_error(d, d_hat_m)
			mae_d = losses.mean_absolute_error(d, d_hat_m)
			loss_d = mae_d + rmse_d
			rmse_neg_d = losses.mean_squared_error(d, x_hat_m)
			mae_neg_d = losses.mean_absolute_error(d, x_hat_m)
			loss_neg_d = mae_neg_d + rmse_neg_d
		
			total_loss = K.abs(loss_d - 0.01*loss_neg_d)
			return total_loss

		# Define loss
		def stft_losses_NMF_x(x, _):
			rmse_x = losses.mean_squared_error(x, x_NMF)
			mae_x = losses.mean_absolute_error(x, x_NMF)
			loss_x = mae_x + rmse_x
			rmse_neg_x = losses.mean_squared_error(x, d_NMF)
			mae_neg_x = losses.mean_absolute_error(x, d_NMF)
			loss_neg_x = mae_neg_x + rmse_neg_x

			total_loss = K.abs(loss_x - 0.01*loss_neg_x)
			return total_loss

		# Define loss
		def stft_losses_NMF_d(d, _):
			rmse_d = losses.mean_squared_error(d, d_NMF)
			mae_d = losses.mean_absolute_error(d, d_NMF)
			loss_d = mae_d + rmse_d
			rmse_neg_d = losses.mean_squared_error(d, x_NMF)
			mae_neg_d = losses.mean_absolute_error(d, x_NMF)
			loss_neg_d = mae_neg_d + rmse_neg_d
		
			total_loss = K.abs(loss_d - 0.01*loss_neg_d)
			return total_loss

		def stft_mag_metric(x, x_hat):
			metric = losses.mean_absolute_error(x, x_hat)

			return metric / BATCH_SIZE    
		
		model_save = TAUnet
		if ngpu > 1:
			TAUnet = multi_gpu_model(TAUnet, gpus=ngpu)
		TAUnet.compile( loss= {'Out_x_hat' : stft_losses_dec_x,
							   'Out_d_hat' : stft_losses_dec_d,
							   'Out_x_NMF' : stft_losses_NMF_x,
							   'Out_d_NMF' : stft_losses_NMF_d},
						loss_weights = {'Out_x_hat' : 1.0,
										'Out_d_hat' : 1.0,
										'Out_x_NMF' : 0.5,
										'Out_d_NMF' : 0.1},
						optimizer='adam',
						#optimizer=optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
						metrics=[stft_mag_metric] )
		TAUnet.summary()
		model = TAUnet
		encoder = TAUnet_encoder
		decoder = TAUnet_decoder

	return model, encoder, decoder, model_save


# Define loss
def unet_loss(x, x_hat):
	mae_loss = K.mean(K.abs(x - x_hat), axis=-1)
	return mae_loss

# spectrogram based Unet architecture
def conv_bat_relu(X,filters, kernel_size, strides):
	out = Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer='he_normal', padding='same')(X)
	out = BatchNormalization(axis=-1)(out)
	out = LeakyReLU(0.2)(out)
	out = MaxPooling2D(pool_size=strides)(out)
	out = Dropout(0.2)(out)
	return out

# spectrogram based Unet architecture
def deconv_bat_relu(X,filters, kernel_size, strides):
	out = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(X)
	out = BatchNormalization(axis=-1)(out)
	out = ReLU()(out)
	out = Dropout(0.2)(out)
	return out

def plot_history(history, fig_name):
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error')
	plt.plot(history.epoch, np.array(history.history['loss']),
			 label='Train Loss')
	plt.plot(history.epoch, np.array(history.history['val_loss']),
			 label = 'Val loss')
	plt.legend()
	#plt.xlim(left=10)
	#plt.ylim([0, ])
	plt.savefig(fig_name + '.png', bbox_inches='tight')
	#plt.show()
