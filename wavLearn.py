import numpy as np
import os
from os.path import isfile, join
import soundfile as sf
import random
from scipy import signal as sig
from scipy.io import savemat
import resampy
import pickle
import wavPlot
import nn_model
import h5py
import subfuncs as F
import matplotlib.pyplot as plot
import random
	
import tensorflow as tf
from keras.callbacks import History, ModelCheckpoint
from sparse_nmf import sparse_nmf
from data_gen_func import DataGenerator
import time

def train_model_NMF(event_name, model_type, r, param):

	#Set of parameters
	t_batch			=param['nmf']['t_batch']
	max_iter		=param['nmf']['max_iter']
	random_seed		=param['nmf']['random_seed']
	sparsity		=param['nmf']['sparsity']
	conv_eps		=param['nmf']['conv_eps']
	cf				=param['nmf']['cf']
	display			=param['nmf']['display']
	cost_check		=param['nmf']['cost_check']

	PATH_MODEL = "./model_NMF/" + event_name + "." + model_type
	
	if os.path.isfile(PATH_MODEL + ".basis") :
		TRAIN_FLAG = 0
	else:    
		TRAIN_FLAG = 1

	if TRAIN_FLAG:   
	#----Load Features-----

		#Load feature info
		PATH_PICKLE = "./feat/" + event_name+"/data.pickle"
		with open(PATH_PICKLE, 'rb') as f_event_info:
			[event_name, DB_path, DB_read_seed, num_file_DB, feat_type, \
			fs, flen, foverlap, nFFT, DC_Bin, feat_len] = pickle.load(f_event_info)

		#load features (Train)
		event_path = "./feat/" + event_name +'/'+feat_type
		PATH_FEAT_TRAIN = event_path + '/dev/'

		flist = []
		for root, dirs, files in os.walk(PATH_FEAT_TRAIN, topdown=False):
			if not root[-3:] == 'bak':
				for name in files:
					flist.append(os.path.join(root, name))
		np.random.shuffle(flist)
		
		nb_feat = 0 
		i = 0
		X_train = []
		while i < len(flist):
			f_event_feat_train = open(flist[i], 'rb')
			X_sub = np.load(f_event_feat_train)
			[t, f] = X_sub.shape
			X_train.append(X_sub)
			nb_feat += t
			i += 1
		X_train = np.concatenate([*X_train],axis=0)
		X_train = np.transpose(X_train)

		#chop big feat seq into batch to prevent memory overflow during NMF
		B = int(nb_feat / t_batch)
		X_train = np.reshape(X_train[:,:B*t_batch], (f, t_batch, B))

		print("Finished loading training dataset")

		#----Train the model----
		#Full Batch Training Mode
		for b in range(0,B):
			if b == 0:
				init_w=None
				init_h=None
			else:
				init_w = w
				init_h = None

			if model_type == 'SNMF_FULLBATCH':
				W_IND= range(0,r)
				H_IND = range(0,r)
				[w, h, obj] = sparse_nmf(X_train[:,:,b], max_iter, random_seed, sparsity, conv_eps, cf, \
										init_w=init_w, init_h=init_h, w_ind=W_IND, h_ind=H_IND, r=r, display=display, cost_check=cost_check)
				

			#Full Batch Training with Nonnegative Matrix Underapproximation (NMU)
			if model_type == 'SNMF_FULLBATCH_NMU':
				STEP_SIZE = 5
				GO_THR =0.015
				for r in range(STEP_SIZE, r, STEP_SIZE):
					W_IND= range(r-STEP_SIZE,r) #Fix previously learned basis
					H_IND = range(0,r) #While refresh activations each time

					if r == STEP_SIZE:
						[w, h, obj] = sparse_nmf(X_train[:,:,b], max_iter, random_seed, sparsity, conv_eps, cf, \
												init_w=None, init_h=None, w_ind=W_IND, h_ind=None, r=r, display=display, cost_check=cost_check)  
						obj_diff = 0
					else:
						obj_prev = obj_now    
						[w, h, obj] = sparse_nmf(X_train[:,:,b], max_iter, random_seed, sparsity, conv_eps, cf, \
												init_w=w, init_h=None, w_ind=W_IND, h_ind=None, r=r, display=display, cost_check=cost_check)  
						obj_now = obj[-1]

						obj_diff_now = (obj_prev - obj_now) / obj_prev
						print('obj difference rate: {}\n'.format(obj_diff_now))

						obj_diff = np.append(obj_diff, obj_diff_now)
						if obj_diff_now < GO_THR:
							break

					obj_now = obj[-1]
					
					#Check Basis by Images
					#wavPlot.waveform(obj, PATH_MODEL + "_R{}.cost".format(r))
					#wavPlot.contour(w, PATH_MODEL + "_R{}.w".format(r))

				#Check learning curve for each trial
				wavPlot.waveform(obj_diff, PATH_MODEL + ".NMU.cost")
			
		#Check Basis by Images
		wavPlot.contour(w, PATH_MODEL + "_R{}.w".format(r))
		wavPlot.waveform(obj, PATH_MODEL + "_R{}.cost".format(r))
		
		#----Save model and its history fig----
		#with open(PATH_MODEL, 'wb') as f_basis:
		#    np.save(f_basis, w)
		#with open(PATH_MODEL_ACTIVATION, 'wb') as f_activation:
		#    np.save(f_activation, h)
		np.savetxt(PATH_MODEL + ".basis", w, delimiter='   ', fmt='%.7e')
		np.savetxt(PATH_MODEL + ".activation", h, delimiter='   ', fmt='%.7e')


def train_model_GC(input_name, target_name, noise_name, interf_name, model_type, PATH_AUXMODEL_X, PATH_AUXMODEL_D, param):

	if param['nn']['cpu_mode'] == 1:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

	#Mixed noise ratio
	SCALE_COND = [0.5, 0.25, 0.125, 0.0625, 0.0442] #2m, 4m, 8m, 16m, 20m
	SCALE_COND_INTERF = [0]#[0.03125, 0.015625] #16m, 32m
	
	#Load feature info
	PATH_PICKLE = "./feat/" + input_name+"/data.pickle"
	with open(PATH_PICKLE, 'rb') as f_event_info:
		[input_name, DB_path, DB_read_seed, num_file_DB, feat_type, \
		fs, flen, foverlap, nFFT, DC_Bin, feat_len] = pickle.load(f_event_info)

	PATH_MODEL_CHECKPOINT = "./model/" + input_name + "." + model_type + ".best.model"
	PATH_MODEL_CHECKPOINT_ENC = "./model/" + input_name + "." + model_type + ".best.model.enc"
	PATH_MODEL_CHECKPOINT_DEC = "./model/" + input_name + "." + model_type + ".best.model.dec"

	#DB pathes
	DB_path_in = "./feat/" + input_name +'/'+feat_type
	DB_path_out1 = "./feat/" + target_name +'/'+feat_type
	DB_path_out2 = "./feat/" + noise_name +'/'+feat_type

	
	F.create_folder('model')

	#Check training or not
	if os.path.isfile(PATH_MODEL_CHECKPOINT_ENC+".data-00000-of-00001") :
		TRAIN_FLAG = 0
	else:    
		TRAIN_FLAG = 1
		
	if TRAIN_FLAG:

		if model_type == 'TAU-Net':
			FEAT_SHAPE = (param['nn']['timestep'], feat_len-1, 1)

			#----Define model structure then compile-----
			#----Load NMF Dictionary----
			B_x = np.loadtxt(PATH_AUXMODEL_X)
			B_d = np.loadtxt(PATH_AUXMODEL_D)
			B_x = np.reshape(B_x[1:,:], (1,512,64))
			B_d = np.reshape(B_d[1:,:], (1,512,64))
			[model, encoder, decoder, model_save] = nn_model.build_model_dae(FEAT_SHAPE, model_type, B_x, B_d, 1, param['nn']['timestep'], param['nn']['ngpu'])
			with open(PATH_MODEL_CHECKPOINT+'.json', 'w') as f:
				f.write(model_save.to_json())
			with open(PATH_MODEL_CHECKPOINT_ENC+'.json', 'w') as f:
				f.write(encoder.to_json())
			with open(PATH_MODEL_CHECKPOINT_DEC+'.json', 'w') as f:
				f.write(decoder.to_json())
				
			[model, encoder, decoder, model_save] = nn_model.build_model_dae(FEAT_SHAPE, model_type, B_x, B_d, param['nn']['batch_size'], param['nn']['timestep'], param['nn']['ngpu'])
			
			##----Begin training----
			
			# Load data Generators for training
			train_gen_val = DataGenerator(DB_path_in+'/dev', DB_path_out1+'/dev', DB_path_out2+'/dev', 
												batch_size=param['nn']['batch_size'],seq_len=param['nn']['timestep'], feat_len=feat_len-1,
												n_channels=1, shuffle=True, per_file=False)

			valid_gen_val = DataGenerator(DB_path_in+'/eval', DB_path_out1+'/eval', DB_path_out2+'/eval', 
												batch_size=param['nn']['batch_size'],seq_len=param['nn']['timestep'], feat_len=feat_len-1,
												n_channels=1, shuffle=False, per_file=False)
			# Load data generator for testing
			_, data_out = train_gen_val.get_data_sizes()
			y_oracle, x_oracle, d_oracle = collect_test_outputs(valid_gen_val, data_out, param['nn']['quick_test'])

			best_mae_metric = 99999
			best_epoch = -1
			patience_cnt = 0
			tr_loss = np.zeros(param['nn']['nb_epochs'])
			val_loss = np.zeros(param['nn']['nb_epochs'])
			mae_metric = np.zeros((param['nn']['nb_epochs'], 2))
			nb_epoch = 30 if param['nn']['quick_test'] else param['nn']['nb_epochs']

			# start training
			for epoch_cnt in range(nb_epoch):
				start = time.time()
			
				# train once per epoch
				hist = model.fit_generator(
					generator=train_gen_val.generate(is_eval=False),
					steps_per_epoch=50 if param['nn']['quick_test'] else train_gen_val.get_total_batches(),
					validation_data=valid_gen_val.generate(is_eval=True),
					validation_steps=50 if param['nn']['quick_test'] else valid_gen_val.get_total_batches(),
					epochs=param['nn']['epochs_per_fit'],
					verbose=1
				)
				tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
				val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]
			
				# predict once per epoch
				out_val = model.predict_generator(
					generator=valid_gen_val.generate(is_eval=True),
					steps=2 if param['nn']['quick_test'] else valid_gen_val.get_total_batches(),
					verbose=1
				)
				x_h = out_val[0]
				d_h = out_val[1]
				x_nmf = out_val[2]
				d_nmf = out_val[3]

				# Calculate the metrics
				mae_metric[epoch_cnt, 0] = np.mean(np.abs(x_h - x_oracle))
				mae_metric[epoch_cnt, 1] = np.mean(np.abs(d_h - d_oracle))
				print('===Validation metric(x) : {} ==='.format(np.mean(np.abs(x_h - x_oracle))))
				print('===Validation metric(d) : {} ==='.format(np.mean(np.abs(d_h - x_oracle))))

				#Visualize intermediate results in spectrogram
				wavPlot.spectrogram_batch(y_oracle,'figs/{}y'.format(epoch_cnt))
				wavPlot.spectrogram_batch(x_h,'figs/{}x_h'.format(epoch_cnt))
				wavPlot.spectrogram_batch(x_nmf,'figs/{}x_nmf'.format(epoch_cnt))
				wavPlot.spectrogram_batch(x_oracle,'figs/{}x_oracle'.format(epoch_cnt))
				wavPlot.spectrogram_batch(d_h,'figs/{}d_h'.format(epoch_cnt))
				wavPlot.spectrogram_batch(d_nmf,'figs/{}d_nmf'.format(epoch_cnt))
				wavPlot.spectrogram_batch(d_oracle,'figs/{}d_oracle'.format(epoch_cnt))
				print('==={}-th epoch figure was saved==='.format(epoch_cnt))

				## Visualize the metrics with respect to param['nn']['epochs_per_fit']
				#plot_functions(unique_name, tr_loss, val_loss, sed_metric, doa_metric, seld_metric)
			
				patience_cnt += 1
				if np.mean(mae_metric[epoch_cnt,:],axis=-1) < best_mae_metric:
					best_mae_metric = np.mean(mae_metric[epoch_cnt,:],axis=-1)
					best_epoch = epoch_cnt
					encoder.save_weights(PATH_MODEL_CHECKPOINT_ENC)
					decoder.save_weights(PATH_MODEL_CHECKPOINT_DEC)    
					model.save_weights(PATH_MODEL_CHECKPOINT)   
					patience_cnt = 0
			
				print(
					'epoch_cnt: %d, time: %.2fs, tr_loss: %.2f, val_loss: %.2f, '
					'MAE_speech: %.2f, MAE_noise: %.2f, '
					'best_seld_score: %.2f, best_epoch : %d\n' %
					(
						epoch_cnt, time.time() - start, tr_loss[epoch_cnt], val_loss[epoch_cnt],
						mae_metric[epoch_cnt, 0], mae_metric[epoch_cnt, 1],
						best_mae_metric, best_epoch
					)
				)

				if patience_cnt == param['nn']['patient']:
					print('patient level reached. finishing training...')
					break
			
			#avg_scores_val.append([sed_metric[best_epoch, 0], sed_metric[best_epoch, 1], doa_metric[best_epoch, 0],
			#					doa_metric[best_epoch, 1], best_seld_metric])
			#print('\nResults on validation split:')
			#print('\tUnique_name: {} '.format(unique_name))
			#print('\tSaved model for the best_epoch: {}'.format(best_epoch))
			#print('\tSELD_score: {}'.format(best_seld_metric))
			#print('\tDOA Metrics: DOA_error: {}, frame_recall: {}'.format(doa_metric[best_epoch, 0],
			#															doa_metric[best_epoch, 1]))
			#print('\tSED Metrics: ER_overall: {}, F1_overall: {}\n'.format(sed_metric[best_epoch, 0],
			#															sed_metric[best_epoch, 1]))
				      


def collect_test_outputs(_data_gen_test, _data_out, quick_test):
	# Collecting ground truth for test data
	nb_batch = 2 if quick_test else _data_gen_test.get_total_batches()

	batch_size = _data_out[0][0]
	y = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2], _data_out[0][3]))
	x = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2], _data_out[0][3]))
	d = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[1][2], _data_out[0][3]))

	print("nb_batch in test: {}".format(nb_batch))
	cnt = 0
	for oracle_in, oracle_out in _data_gen_test.generate(is_eval=True):
		y[cnt * batch_size:(cnt + 1) * batch_size, :, :] = oracle_in
		x[cnt * batch_size:(cnt + 1) * batch_size, :, :] = oracle_out[0]
		d[cnt * batch_size:(cnt + 1) * batch_size, :, :] = oracle_out[1]
		cnt = cnt + 1
		if cnt == nb_batch:
			break
	return y, x, d


def plot_functions(fig_name, _tr_loss, _val_loss, _mae_loss):

	plot.figure()
	nb_epoch = len(_tr_loss)
	plot.subplot(211)
	plot.plot(range(nb_epoch), _tr_loss, label='train loss')
	plot.plot(range(nb_epoch), _val_loss, label='val loss')
	plot.legend()
	plot.grid(True)

	plot.subplot(212)
	plot.plot(range(nb_epoch), _mae_loss[:, 0], label='X mae')
	plot.plot(range(nb_epoch), _mae_loss[:, 1], label='D mae')
	plot.legend()
	plot.grid(True)

	plot.savefig(fig_name)
	plot.close()
