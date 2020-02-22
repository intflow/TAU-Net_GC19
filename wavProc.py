import numpy as np
import os
import random
from scipy import signal as sig
from scipy.fftpack import fft, ifft
from scipy.io import savemat
import resampy
import pickle
import wavPlot
import h5py

from tensorflow import keras
from keras.callbacks import History, ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
from sparse_nmf import sparse_nmf
from sig_subfuncs import mat_squeeze
from sig_subfuncs import st_sparsity
from sig_subfuncs import ReLU
import nn_model
			
#----TAU-Net (Noisy -> TAU-Enc -> NMF-refine -> TAU-Dec -> Clean)
def TAU_Net(y, fs, models, param, fname, PROC_METHOD, PATH_SUBMODEL_X, PATH_SUBMODEL_D):
	
	#----Algorithm specific parameters
	FRAME_SIZE = param['val']['FRAME_SIZE'] #unoverlapped unit frame size
	OVERLAP_RATIO = param['val']['OVERLAP_RATIO'] #No. of overlapped frame
	FLR = param['val']['FLR'] #Nonzero flooring value
	BLOCK_F = param['val']['BLOCK_F'] #Freqency bin # of unit spectrogram block feature
	BLOCK_T = param['val']['BLOCK_T'] #time frame # of unit spectrogram block feature
	HYBRID_MODEL = param['val']['HYBRID_MODEL'] #0: Use only TAU model, 1: Use Encoder -> NMF TA-Refine -> Decoder
	MAX_ITER = param['val']['MAX_ITER'] #NMF Iteration
	FEAT_TYPE = param['val']['FEAT_TYPE'] #Feature type
	#==Soft Mask estimation
	MMSE = param['val']['MMSE'] #0: Wiener Gain, 1: 
	BETA_D = param['val']['BETA_D'] #Default noise reduction gain
	BETA_MAX = param['val']['BETA_MAX'] #Maximum noise reduction gain by Activation SNR
	ALPHA_D = param['val']['ALPHA_D'] #Noise smoothing factor
	ALPHA_ETA = param['val']['ALPHA_ETA'] #Target smoothing factor
	ETA_FLR = param['val']['ETA_FLR']
	FLR = param['val']['FLR'] #Nonzero flooring value
	FILT_FLR = param['val']['FILT_FLR'] #prevent zero after noise subtraction
	FILTER_POW = param['val']['FILTER_POW'] #Gain filter's Power scale: lower -> pron to SAR
	SCALE_FIT = param['val']['SCALE_FIT']

	#==Spectro-temporal Sparsity check
	ST_SPARSITY = param['val']['ST_SPARSITY']
	THETA_R = param['val']['THETA_R']
	K_q = param['val']['K_q']
	I_q = param['val']['I_q']
	GAMMA = param['val']['GAMMA']
	DEFAULT_SPARSITY = param['val']['DEFAULT_SPARSITY']
	#==DPSC options
	DPSC = param['val']['DPSC']
	#==Adaptive Noise Training
	ADAPT_TRAIN_D = param['val']['ADAPT_TRAIN_D'] #Turn On/Off Noise Training
	MAX_ITER_ONL = param['val']['MAX_ITER_ONL'] #Maximum NMF Iteration on Noise Training
	AR_UP = param['val']['AR_UP'] #define Ax and Ad ratio for noise dictionary update (Lower: Update frequently, Higher: Update rarely)
	INIT_D_LEN = param['val']['INIT_D_LEN'] #Initialization Frame
	M_A = param['val']['M_A'] #Stacked no. of block for noise training
	r_a = param['val']['r_a'] #Noise update ranks
	KAPPA_INV = param['val']['KAPPA_INV']

	#==GC_Drone
	try:
		CUT_FREQ = param['GCtask3']['CUT_FREQ']
	except:
		CUT_FREQ = 0

	WINDOW_SIZE = FRAME_SIZE * OVERLAP_RATIO
	MAG_SIZE = int(WINDOW_SIZE / 2 + 1)
	K = MAG_SIZE
	UP_STEP = INIT_D_LEN + M_A #Noise Training Cycle

	#==Set Keras models
	[model, encoder, decoder]= models

	#==Load NMF Auxiliary models
	B_x = np.loadtxt(PATH_SUBMODEL_X)
	B_d = np.loadtxt(PATH_SUBMODEL_D)
	(k,r_x)=B_x.shape
	(k,r_d)=B_d.shape
	B_a = B_d[:,r_d-r_a:] * 0.9
	r = r_x + r_d + r_a

	#----buffer init (OLA)
	t_idx = 0
	y = np.array(y, dtype='int16')
	#----compensate block delay----
	pad = np.zeros(FRAME_SIZE * (BLOCK_T-1))
	y = np.concatenate([y,pad]) 
	#------------------------------
	y_len_org = len(y)
	y = np.concatenate([y, np.zeros(FRAME_SIZE * OVERLAP_RATIO)])
	x_hat = np.zeros(len(y), dtype='int16')
	y_buf = np.zeros((OVERLAP_RATIO, WINDOW_SIZE), dtype='int16')
	y_win = np.zeros(WINDOW_SIZE, dtype='int16')
	x_hat_win = np.zeros((OVERLAP_RATIO, WINDOW_SIZE), dtype='int16')
	OLA_win = np.sqrt(sig.hanning(WINDOW_SIZE, sym=False)) #sqrt hanning
	OLA_gain = np.sum(OLA_win ** 2)

	#----buffer init(algorithm)
	lambda_d = np.zeros([MAG_SIZE,1], dtype='float')
	X_out = np.zeros([MAG_SIZE,1], dtype='float')
	frm_cnt = 0
	r_blk = np.zeros((I_q, MAG_SIZE))

	#STFT Batch buffers
	r_blk = np.zeros((I_q, MAG_SIZE))
	Y_blk_cpx = np.ones((BLOCK_T,WINDOW_SIZE), dtype='complex128')
	Y_blk = np.ones((BLOCK_T,BLOCK_F))
	X_blk = np.ones((BLOCK_T,BLOCK_F)) + 0.000001
	D_blk = np.ones((BLOCK_T,BLOCK_F)) + 0.000001
	A_x_blk = np.zeros((r_x, BLOCK_T)) + 0.000001
	A_d_blk = np.zeros((r_d, BLOCK_T)) + 0.000001
	A_a_blk = np.zeros((r_a, BLOCK_T)) + 0.000001
	blk_idx = 0
	while(t_idx + FRAME_SIZE < len(y)):
		y_t = y[t_idx:t_idx+FRAME_SIZE]
		y_buf[0, 0:WINDOW_SIZE-FRAME_SIZE] = y_buf[0, FRAME_SIZE:WINDOW_SIZE] 
		y_buf[0, WINDOW_SIZE-FRAME_SIZE:WINDOW_SIZE] = y_t
		y_win = y_buf[0,:].astype('float') * OLA_win
		#y_buf[0, :] = 32000.0 * np.ones(WINDOW_SIZE) * sig.hanning(WINDOW_SIZE, sym=False)

		#----Begin Frame-Wise Process
		Y = fft(y_win / OLA_gain, WINDOW_SIZE)
		Y = Y.reshape([WINDOW_SIZE, 1])

		#----GCtask3 options
		if CUT_FREQ > 0:
			cut_bin = int(CUT_FREQ/fs * MAG_SIZE)
			Y[:cut_bin] *= 0.00001
			Y[-cut_bin:] *= 0.00001


		Y_mag = np.abs(Y)[0:MAG_SIZE]
		Y_rad =	 np.reshape(np.angle(Y), (WINDOW_SIZE, 1))

		#----STFT block to spectrum process
		Y_blk_cpx[:-1,:] = Y_blk_cpx[1:,:]
		Y_blk_cpx[-1,:] = np.squeeze(Y)
		Y_blk[:-1,:] = Y_blk[1:,:] #shift
		Y_blk[-1,:] = np.squeeze(Y_mag[1:]) #drop lowest freq bin
		Y_cpx = Y_blk_cpx[0,:]
		Y_cpx = Y_cpx.reshape([WINDOW_SIZE, 1])
		Y_feat = Y_blk[0,:]
		X_est = X_blk[blk_idx,:] #Should update index since block does not change until block-wise inference
		D_est = D_blk[blk_idx,:] + FLR #Should update index since block does not change until block-wise inference
		A_x = np.reshape(A_x_blk[:,blk_idx], (r_x, 1))
		A_d = np.reshape(A_d_blk[:,blk_idx], (r_d, 1))
		A_a = np.reshape(A_a_blk[:,blk_idx], (r_a, 1))

		Y_feat = np.reshape(np.insert(Y_feat, 0, 0), (K, 1))
		X_est = np.reshape(np.insert(X_est, 0, 0), (K, 1))
		D_est = np.reshape(np.insert(D_est, 0, 1), (K, 1))
		blk_idx += 1

		#----Analyze spectro-temporal sparsity
		if ST_SPARSITY == 1:
			(Q_sparsity, r_blk) = st_sparsity(X_est**2, D_est**2, r_blk, THETA_R, K, K_q, I_q, GAMMA)
		else:
			Q_sparsity = np.ones([MAG_SIZE, 1]) * DEFAULT_SPARSITY

		#----Set Filter
		#1)Conventional Wiener Filter
		if MMSE == 0:
			G = X_est**2 * Q_sparsity / (X_est**2 * Q_sparsity + D_est**2)
			#G = X_est**2 * Q_sparsity / ((X_est * Q_sparsity + D_est) ** 2)
			
		#2)MMSE filter 
		#IS16, Jeon, Modified for TAU-Net
		A_x_mag = np.average(A_x, axis=0)
		A_d_mag = np.average(A_d, axis=0) + np.average(A_a, axis=0)
		beta = 20 * np.log10(A_d_mag / A_x_mag)
		if (beta < BETA_D):
			beta = BETA_D
		elif (beta >= BETA_MAX):
			beta = BETA_MAX
		
		if MMSE == 1:
			if frm_cnt <= INIT_D_LEN:
				lambda_d = ((1-ALPHA_D) * lambda_d + (ALPHA_D) * D_est**2 + 1)
				eta = ((1-ALPHA_ETA)*X_out + (ALPHA_ETA) * X_est * Q_sparsity) / (lambda_d + FLR)
			else:
				lambda_d = ALPHA_D * lambda_d + (1 - ALPHA_D) * D_est**2 * beta
				eta = ((ALPHA_ETA)*X_out + (1-ALPHA_ETA) * X_est**2 * Q_sparsity) / (lambda_d + FLR)

			eta = eta + ETA_FLR
			G = eta / (eta + 1)
			G = np.minimum(G, 1.0)
			G = np.power(G,FILTER_POW)
			G *= SCALE_FIT #Boost 1dB, by heuristic
		X_out = Y_feat.reshape(MAG_SIZE,1) * G

		#3)Dynamic Phase Spectrum Compensation
		if DPSC > 0:
			if DPSC == 1: #Constant method by K.Wojcicki
				#phase_lambda = np.ones((MAG_SIZE,1)) * 5 #constant PSC
				phase_lambda = (D_est) * 3.74 #constant PSC
				X_hat_m = (Y_feat.reshape(MAG_SIZE,1))
			elif DPSC == 2: #Dynamic PSC by kmjeon
				Interf = (1-Q_sparsity) * (X_est) - Q_sparsity * (D_est)
				Interf = np.maximum(Interf, FLR)
				D = (D_est) + Interf
				lSNR = np.maximum(X_est / D_est, THETA_R)
				phase_lambda = (1 / lSNR) * D
				X_hat_m = (Y_feat.reshape(MAG_SIZE,1)) * np.sqrt(G)

				###fwwrite for Sparsity vs LSNR relationship in Fig.3 of SPSC
				##flists = fname.split('_')
				##if flists[1] == '5' and flists[2] == 'dliv':
				##    with open("sparse_vs_LSNR_DRNN_dliv.txt", "a") as myfile:
				##        for i in range(0,len(Q_sparsity)):
				##            if 1/lSNR[i,0] < 4.0:
				##                myfile.write("{0:.4f} {1:.4f}\n".format(Q_sparsity[i,0], 1/lSNR[i,0]))
				##
				##if flists[1] == '5' and flists[2] == 'nriv':
				##    with open("sparse_vs_LSNR_DRNN_nriv.txt", "a") as myfile:
				##        for i in range(0,len(Q_sparsity)):
				##            if 1/lSNR[i,0] < 4.0:
				##                myfile.write("{0:.4f} {1:.4f}\n".format(Q_sparsity[i,0], 1/lSNR[i,0]))
				##
				##if flists[1] == '5' and flists[2] == 'pcaf':
				##    with open("sparse_vs_LSNR_DRNN_pcaf.txt", "a") as myfile:
				##        for i in range(0,len(Q_sparsity)):
				##            if 1/lSNR[i,0] < 4.0:
				##                myfile.write("{0:.4f} {1:.4f}\n".format(Q_sparsity[i,0], 1/lSNR[i,0]))
				##            
				##if flists[1] == '5' and flists[2] == 'tmet':
				##    with open("sparse_vs_LSNR_DRNN_tmet.txt", "a") as myfile:
				##        for i in range(0,len(Q_sparsity)):
				##            if 1/lSNR[i,0] < 4.0:
				##                myfile.write("{0:.4f} {1:.4f}\n".format(Q_sparsity[i,0], 1/lSNR[i,0]))

			phase_lambda = np.concatenate([phase_lambda,  -1*np.flipud(phase_lambda[1:MAG_SIZE-1])])
			X_hat_mod = Y_cpx + phase_lambda
			X_hat_p = np.angle(X_hat_mod)
			X_hat = np.concatenate([X_hat_m,  np.flipud(X_hat_m[1:MAG_SIZE-1])]) * \
					np.exp(1j*X_hat_p)
		else:	
			G = np.sqrt(G)
			X_hat = Y_cpx * np.concatenate([G,	np.flipud(G[1:MAG_SIZE-1])])

		x_hat_win[0,:] = ifft(X_hat.T * OLA_gain, WINDOW_SIZE)
		x_hat_win[0,:] = x_hat_win[0,:] * OLA_win

		#----Reconstruct OLA
		x_hat_t = np.zeros(FRAME_SIZE)
		for i in range(OVERLAP_RATIO-1, -1, -1):
			x_hat_t = x_hat_t + x_hat_win[i, i * FRAME_SIZE : (i+1) * FRAME_SIZE]

		x_hat_t = x_hat_t / (OVERLAP_RATIO / 2)
		if t_idx >= FRAME_SIZE * (OVERLAP_RATIO-1):
			x_hat[t_idx - FRAME_SIZE * (OVERLAP_RATIO-1):t_idx - FRAME_SIZE * (OVERLAP_RATIO-2)] = x_hat_t.astype('int16')

		#4)Online Noise Learning
		if ADAPT_TRAIN_D == 1:
			if ((frm_cnt == UP_STEP) or (frm_cnt < INIT_D_LEN)):
				if (frm_cnt < INIT_D_LEN): #Noise only region
					A_x_mag = 0
				
				A_x_mag_comp = np.ones(r_a) * A_x_mag
				Q_control = (1 - np.mean(Q_sparsity) ** KAPPA_INV) * AR_UP
				if (Q_control * A_d_mag > A_x_mag) or (frm_cnt < INIT_D_LEN):
					if frm_cnt < INIT_D_LEN:
						D_ref = Y_feat.reshape([MAG_SIZE,1]) + FLR
					else:
						M_ref = 1 - G
						D_ref = Y_feat.reshape([MAG_SIZE,1]) * M_ref.reshape([MAG_SIZE,1])
					
					if frm_cnt <= BLOCK_T:
						lambda_d_blk = np.tile(D_ref, [1, M_A])
						A_a_blk_ONL = np.tile(A_a, [1, M_A])
					else:
						lambda_d_blk = np.concatenate([lambda_d_blk[:,1:M_A], D_ref], axis = 1)
						A_a_blk_ONL = np.concatenate([A_a_blk_ONL[:,1:M_A], A_a], axis = 1)
						##wavPlot.spectrogram(lambda_d_blk, None)  

					if frm_cnt > BLOCK_T:
						r_up = (Q_control * np.mean(A_a_blk_ONL, 1)) > A_x_mag_comp
						r_up_inv = 1 - r_up

						A_a_blk_up = A_a_blk_ONL
						A_a_blk_up, r_a_up = mat_squeeze(A_a_blk_up, r_up, 0)
						B_A_up = B_a[:,0:r_a]
						[B_A_up, _] = mat_squeeze(B_A_up, r_up, 1)
						B_A_rem = B_a[:,0:r_a]
						[B_A_rem, _]  = mat_squeeze(B_A_rem, r_up_inv, 1)

						if (r_a_up > 0):
							W_IND = range(0,r_a)
							H_IND = range(0,0)
							#init_w = B_D_up #pretrained basis from training set
							init_h = A_a 
							[B_A_up, _, _] = sparse_nmf(lambda_d_blk, max_iter=MAX_ITER_ONL, random_seed=1, sparsity=5.00, conv_eps=0.0001, cf='kl', \
													init_w=None, init_h=None, w_ind=W_IND, h_ind=H_IND, r=r_a, display=0, cost_check=1)
							
						#if r_a == r_d:
						#	 B_a = B_A_up
						#else:
						if r_a_up == r_a:
							B_a = B_A_up
						else:
							B_a = np.concatenate([B_A_rem[:,0:r_a-r_a_up], B_A_up[:,0:r_a_up]], axis =1 )
						##wavPlot.spectrogram(B_a, None)	
					
			if frm_cnt < UP_STEP:
				frm_cnt = frm_cnt + 1
			else:
				frm_cnt = INIT_D_LEN

		#index update
		for i in range(OVERLAP_RATIO-1, 0, -1):
			x_hat_win[i, :] = x_hat_win[i-1, :]

		t_idx = t_idx + FRAME_SIZE	
		#----Get Speech and Noise codes from Speech/Noise Encoders
		if blk_idx == BLOCK_T:
			blk_idx = 0
			norm_val = np.max(Y_blk) #Max norm of mag

			#====Feature conversion for TAU-Net====
			#Y_blk_log = np.log10(np.square(Y_blk)+1.0001) #Log10 feature for TAU-Net
			#norm_val_log = np.amax(Y_blk_log) #Max norm log10(mag)
			#tmp = Y_blk_log / (norm_val_log+1)
			#=======================================
			tmp = Y_blk / (norm_val+1)
			_Y = np.reshape(tmp, (1, BLOCK_T, BLOCK_F, 1))
			if HYBRID_MODEL == 0:
				mid_outs = encoder.predict(_Y, batch_size=1)
				[_X, _D] = decoder.predict([_Y, *mid_outs], batch_size=1)
				X_blk = np.squeeze(_X)
				D_blk = np.squeeze(_D)
				X_blk *= (norm_val+1)
				D_blk *= (norm_val+1)
				#====Feature conversion for TAU-Net (back to mag)====
				#X_blk = np.power(10,X_blk) - 1.0001
				#D_blk = np.power(10,D_blk) - 1.0001
				#X_blk = np.sqrt(X_blk)
				#np.sqrt(D_blk)
				#=======================================

			if HYBRID_MODEL > 0:
				#----Activation estimation from TAU-Net Encoder
				mid_outs = encoder.predict(_Y, batch_size=1)
				A_blk = np.squeeze(mid_outs[0])
				A_x_blk = A_blk[:,:r_x].T #(A,T)
				A_d_blk = A_blk[:,r_x:].T
				
				#----SNMF-based activation fitting
				Y_NMF_in = (Y_blk / (norm_val+1)).T + 0.0001
				if ADAPT_TRAIN_D == 1:
					init_w = np.concatenate([B_x[1:,:], B_d[1:,:], B_a[1:,:]], axis=1) #pretrained basis from training set
				else:
					init_w = np.concatenate([B_x[1:,:], B_d[1:,:]], axis=1)
					r = r_x + r_d
					
				W_IND = range(0,0)
				H_IND = range(0,r)
				init_h = np.concatenate([A_x_blk, A_d_blk], axis=0)
				[_, A, _] = sparse_nmf(Y_NMF_in, max_iter=MAX_ITER, random_seed=3, sparsity=5, conv_eps=0.0001, cf='kl', \
										init_w=init_w, init_h=init_h, w_ind=W_IND, h_ind=H_IND, r=r, display=0, cost_check=1)
				A_x_blk = A[0:r_x,:]
				A_d_blk = A[r_x:r_x+r_d,:]
				A_a_blk = A[r_x+r_d:r,:]

				if ADAPT_TRAIN_D == 0:
					A_a_blk = np.zeros((r_a,BLOCK_T)) + FLR
				#init_h = np.concatenate([A_x_blk, A_d_blk], axis=0)
				#wavPlot.spectrogram(init_h.T, None)
				
				Da_multi = B_a[1:,:] @ A_a_blk
				Da_blk = Da_multi.T
				Da_blk *= (norm_val+1)

				if HYBRID_MODEL == 1:
					A_ref = np.concatenate([A_x_blk,A_d_blk],axis=0)
					mid_outs[0] = np.reshape(A_ref, (1,BLOCK_T,r_x+r_d))
					
					#====Conduct activation fine-tuning====
					[_X, _D] = decoder.predict([_Y, *mid_outs], batch_size=1)
					X_blk = np.squeeze(_X)
					D_blk = np.squeeze(_D)
					X_blk *= (norm_val+1)
					D_blk *= (norm_val+1)
					
					##====Feature conversion for TAU-Net (back to mag)====
					#X_blk = np.power(10,X_blk) - 1.0001
					#D_blk = np.power(10,D_blk) - 1.0001
					#X_blk = np.sqrt(X_blk)
					#D_blk = np.sqrt(D_blk)
					##=======================================
						
				if HYBRID_MODEL == 2:
					X_multi = B_x[1:,:] @ A_x_blk
					D_multi = B_d[1:,:] @ A_d_blk
					X_blk = X_multi.T
					D_blk = D_multi.T
					X_blk *= (norm_val+1)
					D_blk *= (norm_val+1)
				
				D_blk += Da_blk

			##wavPlot.spectrogram(X_blk.T, None)
	
	x_hat = x_hat[len(pad):y_len_org] #Compensate block delay
	return x_hat

#---- Load Keras models for signal processing
def load_sep_model(feat_len, timestep, model_type, PATH_MODEL_X, PATH_MODEL_D, PATH_SUBMODEL_X, PATH_SUBMODEL_D):

	if model_type == 'TAU-Net':
		FEAT_SHAPE = (timestep, feat_len-1, 1)
		BATCH_SIZE = 1
		#----Load NMF Dictionary----
		B_x = np.loadtxt(PATH_SUBMODEL_X)
		B_d = np.loadtxt(PATH_SUBMODEL_D)
		B_x = np.reshape(B_x[1:,:].T, (1,512,64))
		B_d = np.reshape(B_d[1:,:].T, (1,512,64))
		B_x = np.tile(B_x, (BATCH_SIZE, 1, 1))
		B_d = np.tile(B_d, (BATCH_SIZE, 1, 1))

		#with open(PATH_MODEL_X+'.json', 'r') as f:
		#	 model = model_from_json(f.read())		  
		#with open(PATH_MODEL_X+'.enc.json', 'r') as f:
		#	 encoder = model_from_json(f.read())
		#with open(PATH_MODEL_X+'.dec.json', 'r') as f:
		#	 decoder = model_from_json(f.read())			
		[_, encoder, decoder, model] = nn_model.build_model_dae(FEAT_SHAPE, model_type, B_x, B_d, BATCH_SIZE, timestep, ngpu = 1)
		#with open(PATH_MODEL_X+'.json', 'w') as f:
		#	f.write(model.to_json())
		#with open(PATH_MODEL_X+'.enc.json', 'w') as f:
		#	f.write(encoder.to_json())
		#with open(PATH_MODEL_X+'.dec.json', 'w') as f:
		#	f.write(decoder.to_json())

		model.load_weights(PATH_MODEL_X)
		encoder.load_weights(PATH_MODEL_X+'.enc')
		decoder.load_weights(PATH_MODEL_X+'.dec')
		models = model, encoder, decoder

	return models

