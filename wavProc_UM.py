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
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

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
	IBM = param['val']['IBM']
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
	B_a = B_d[:,r_d-r_a:]
	r = r_x + r_d + r_a

	#----buffer init (OLA)
	t_idx = 0
	y = np.array(y, dtype='int16')
	ch_num = np.shape(y)[1]

	#----compensate block delay----
	pad = np.zeros((FRAME_SIZE * (BLOCK_T-1),ch_num))
	y = np.concatenate([y,pad],axis=0) 
	#------------------------------
	y_len_org = np.shape(y)[0]
	y = np.concatenate([y, np.zeros((FRAME_SIZE * OVERLAP_RATIO, ch_num))],axis =0)
	x_hat = np.zeros(np.shape(y), dtype='int16')
	y_buf = np.zeros((OVERLAP_RATIO, WINDOW_SIZE, ch_num), dtype='int16')
	y_win = np.zeros(WINDOW_SIZE, dtype='int16')
	x_hat_win = np.zeros((OVERLAP_RATIO, WINDOW_SIZE, ch_num), dtype='int16')
	OLA_win = np.sqrt(sig.hanning(WINDOW_SIZE, sym=False)) #sqrt hanning
	OLA_gain = np.sum(OLA_win ** 2)

	#----buffer init(algorithm)
	lambda_d = np.zeros([MAG_SIZE,ch_num], dtype='float')
	X_out = np.zeros([MAG_SIZE,ch_num], dtype='float')
	G_mc = np.zeros([MAG_SIZE,ch_num], dtype='float')
	G_IBM_mc = np.zeros([MAG_SIZE,ch_num], dtype='int')
	Y_cpx = np.zeros([WINDOW_SIZE,ch_num], dtype='complex64')
	frm_cnt = 0
	
	#STFT Batch buffers
	r_blk = np.zeros((I_q, MAG_SIZE,ch_num))
	Y_blk_cpx = np.ones((BLOCK_T,WINDOW_SIZE,ch_num), dtype='complex128')
	Y_blk = np.ones((BLOCK_T,BLOCK_F,ch_num))
	X_blk = np.ones((BLOCK_T,BLOCK_F,ch_num)) + 0.000001
	D_blk = np.ones((BLOCK_T,BLOCK_F,ch_num)) + 0.000001
	A_x_blk = np.zeros((r_x, BLOCK_T,ch_num)) + 0.000001
	A_d_blk = np.zeros((r_d, BLOCK_T,ch_num)) + 0.000001
	A_a_blk = np.zeros((r_a, BLOCK_T,ch_num)) + 0.000001
	blk_idx = np.zeros(ch_num, dtype='int')

	#VAD indices buffers
	if param['vad']['is_vad'] == 1:
		nb_harmonic = []
		nb_harmonic_delta = []
		vad = []
		nb_harmonic_now_1d = 0

	while(t_idx + FRAME_SIZE < np.shape(y)[0]):

		for ch in range(ch_num):
			y_t = y[t_idx:t_idx+FRAME_SIZE,ch]
			y_buf[0, 0:WINDOW_SIZE-FRAME_SIZE,ch] = y_buf[0, FRAME_SIZE:WINDOW_SIZE,ch] 
			y_buf[0, WINDOW_SIZE-FRAME_SIZE:WINDOW_SIZE,ch] = y_t
			y_win = y_buf[0,:,ch].astype('float') * OLA_win
			#y_buf[0, :] = 32000.0 * np.ones(WINDOW_SIZE) * sig.hanning(WINDOW_SIZE, sym=False)

			#----Begin Frame-Wise Process
			Y = fft(y_win / OLA_gain, WINDOW_SIZE)
			Y = Y.reshape([WINDOW_SIZE, 1])

			Y_mag = np.abs(Y)[0:MAG_SIZE]
			#Y_rad =	 np.reshape(np.angle(Y), (WINDOW_SIZE, 1))

			#----STFT block to spectrum process
			Y_blk_cpx[:-1,:,ch] = Y_blk_cpx[1:,:,ch]
			Y_blk_cpx[-1,:,ch] = np.squeeze(Y)
			Y_blk[:-1,:,ch] = Y_blk[1:,:,ch] #shift
			Y_blk[-1,:,ch] = np.squeeze(Y_mag[1:]) #drop lowest freq bin
			Y_cpx[:,ch] = Y_blk_cpx[0,:,ch]
			#Y_cpx[:,ch] = Y_cpx[:,ch].reshape([WINDOW_SIZE, 1])
			Y_feat = Y_blk[0,:,ch]
			X_est = X_blk[blk_idx[ch],:,ch] #Should update index since block does not change until block-wise inference
			D_est = D_blk[blk_idx[ch],:,ch] + FLR #Should update index since block does not change until block-wise inference
			A_x = np.reshape(A_x_blk[:,blk_idx[ch],ch], (r_x, 1))
			A_d = np.reshape(A_d_blk[:,blk_idx[ch],ch], (r_d, 1))
			A_a = np.reshape(A_a_blk[:,blk_idx[ch],ch], (r_a, 1))

			Y_feat = np.reshape(np.insert(Y_feat, 0, 0), (K, 1))
			X_est = np.reshape(np.insert(X_est, 0, 0), (K, 1))
			D_est = np.reshape(np.insert(D_est, 0, 1), (K, 1))
			blk_idx[ch] += 1

			#----Analyze spectro-temporal sparsity
			if ST_SPARSITY == 1:
				(Q_sparsity, r_blk[:,:,ch]) = st_sparsity(X_est**2, D_est**2, r_blk[:,:,ch], THETA_R, K, K_q, I_q, GAMMA)
			else:
				Q_sparsity = np.ones([MAG_SIZE, 1]) * DEFAULT_SPARSITY

			#----Set Filter
			#Do IBM pre-processing
			if IBM > 0:
				G_IBM = np.zeros(X_est.shape, dtype='int32')
				if np.sum(X_est) != 512.0:
					G_IBM[:param['val']['IBM_BAND']] = X_est[:param['val']['IBM_BAND']] > IBM*D_est[:param['val']['IBM_BAND']]
					G_IBM[param['val']['IBM_BAND']:] = X_est[param['val']['IBM_BAND']:] > D_est[param['val']['IBM_BAND']:]

				X_est = X_est * G_IBM
				D_est = D_est * (1-G_IBM)
				G_IBM_mc[:,ch] = np.squeeze(G_IBM)

			#1)Conventional Wiener Filter
			if MMSE == 0:
				G = X_est**FILTER_POW * Q_sparsity / (X_est**FILTER_POW * Q_sparsity + D_est**FILTER_POW)
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
					lambda_d[:,ch] = ((1-ALPHA_D) * lambda_d[:,ch] + (ALPHA_D) * D_est[:,0]**FILTER_POW + 1)
					eta = ((1-ALPHA_ETA)*X_out[:,ch]**FILTER_POW + (ALPHA_ETA) * X_est[:,0]**FILTER_POW * Q_sparsity[:,0]) / (lambda_d[:,ch] + FLR)
				else:
					lambda_d[:,ch] = ALPHA_D * lambda_d[:,ch] + (1 - ALPHA_D) * D_est[:,0]**FILTER_POW * beta
					eta = ((ALPHA_ETA)*X_out[:,ch]**FILTER_POW + (1-ALPHA_ETA) * X_est[:,0]**FILTER_POW * Q_sparsity[:,0]) / (lambda_d[:,ch] + FLR)

				eta = eta + ETA_FLR
				G = eta / (eta + 1)
				G = np.minimum(G, 1.0)
			X_out[:,ch] = np.squeeze(Y_feat.reshape(MAG_SIZE,1) * G.reshape(MAG_SIZE,1)**(1/FILTER_POW))

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
					X_hat_m = Y_feat.reshape(MAG_SIZE,1) * G**(1/FILTER_POW)

				phase_lambda = np.concatenate([phase_lambda,  -1*np.flipud(phase_lambda[1:MAG_SIZE-1])])
				X_hat_mod = Y_cpx[:,ch] + phase_lambda
				X_hat_p = np.angle(X_hat_mod)
				X_hat = np.concatenate([X_hat_m,  np.flipud(X_hat_m[1:MAG_SIZE-1])]) * \
						np.exp(1j*X_hat_p)
			else:	
				G = G**(1/FILTER_POW)
				G_mc[:,ch] = np.squeeze(G)

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
								init_w = B_A_up #pretrained basis from training set
								init_h = A_a 
								[B_A_up, _, _] = sparse_nmf(lambda_d_blk, max_iter=MAX_ITER_ONL, random_seed=1, sparsity=5.00, conv_eps=0.0001, cf='kl', \
														init_w=init_w, init_h=init_h, w_ind=W_IND, h_ind=H_IND, r=r_a, display=0, cost_check=1)
								
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

			if ch == ch_num-1:
				t_idx += FRAME_SIZE	

			#----Get Speech and Noise codes from Speech/Noise Encoders
			if blk_idx[ch] == BLOCK_T:
				blk_idx[ch] = 0
				norm_val = np.max(Y_blk[:,:,ch]) #Max norm of mag

				tmp = Y_blk[:,:,ch] / (norm_val+1)
				_Y = np.reshape(tmp, (1, BLOCK_T, BLOCK_F, 1))
				if HYBRID_MODEL == 0:
					mid_outs = encoder.predict(_Y, batch_size=1)
					[_X, _D] = decoder.predict([_Y, *mid_outs], batch_size=1)
					X_blk[:,:,ch] = np.squeeze(_X)
					D_blk[:,:,ch] = np.squeeze(_D)
					X_blk[:,:,ch] *= (norm_val+1)
					D_blk[:,:,ch] *= (norm_val+1)
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
					A_x_blk[:,:,ch] = A_blk[:,:r_x].T #(A,T)
					A_d_blk[:,:,ch] = A_blk[:,r_x:].T
					
					#----SNMF-based activation fitting
					Y_NMF_in = (Y_blk[:,:,ch] / (norm_val+1)).T + 0.0001
					if ADAPT_TRAIN_D == 1:
						init_w = np.concatenate([B_x[1:,:], B_d[1:,:], B_a[1:,:]], axis=1) #pretrained basis from training set
					else:
						init_w = np.concatenate([B_x[1:,:], B_d[1:,:]], axis=1)
						r = r_x + r_d
						
					W_IND = range(0,0)
					H_IND = range(0,r)
					init_h = np.concatenate([A_x_blk[:,:,ch], A_d_blk[:,:,ch]], axis=0)
					[_, A, _] = sparse_nmf(Y_NMF_in, max_iter=MAX_ITER, random_seed=3, sparsity=5, conv_eps=0.0000001, cf='kl', \
										   init_w=init_w, init_h=init_h, w_ind=W_IND, h_ind=H_IND, r=r, display=0, cost_check=1)
					A_x_blk[:,:,ch] = A[0:r_x,:]
					A_d_blk[:,:,ch] = A[r_x:r_x+r_d,:]
					A_a_blk[:,:,ch] = A[r_x+r_d:r,:]
					
					if ADAPT_TRAIN_D == 0:
						A_a_blk[:,:,ch] = np.zeros((r_a,BLOCK_T)) + FLR
					#init_h = np.concatenate([A_x_blk[:,:,ch], A_d_blk[:,:,ch]], axis=0)
					#wavPlot.contour(init_h, None)
					
					Da_multi = B_a[1:,:] @ A_a_blk[:,:,ch]
					Da_blk = Da_multi.T
					Da_blk *= (norm_val+1)
					
					if HYBRID_MODEL == 1:
						A_ref = np.concatenate([A_x_blk[:,:,ch].T,A_d_blk[:,:,ch].T],axis=-1) #(T,R)
						mid_outs[0] = np.reshape(A_ref, (1,BLOCK_T,r_x+r_d))
						
						#====Conduct activation fine-tuning====
						[_X, _D] = decoder.predict([_Y, *mid_outs], batch_size=1)
						X_blk[:,:,ch] = np.squeeze(_X)
						D_blk[:,:,ch] = np.squeeze(_D)
						X_blk[:,:,ch] *= (norm_val+1)
						D_blk[:,:,ch] *= (norm_val+1)
						D_blk[:,:,ch] += Da_blk
							
					if HYBRID_MODEL == 2:
						X_multi = B_x[1:,:] @ A_x_blk[:,:,ch]
						D_multi = B_d[1:,:] @ A_d_blk[:,:,ch]
						X_blk[:,:,ch] = X_multi.T
						D_blk[:,:,ch] = D_multi.T
						X_blk[:,:,ch] *= (norm_val+1)
						D_blk[:,:,ch] *= (norm_val+1)
						D_blk[:,:,ch] += Da_blk
					#wavPlot.spectrogram(X_blk[:,:,ch].T * (norm_val+1),'X_blk')
					#wavPlot.spectrogram(D_blk[:,:,ch].T * (norm_val+1),'X_blk')
					#wavPlot.spectrogram(B_x[1:,:] @ A_x_blk[:,:,ch] * (norm_val+1),'X_blk_NMF')
					#wavPlot.spectrogram(B_d[1:,:] @ A_d_blk[:,:,ch] * (norm_val+1),'D_blk_NMF')
					#wavPlot.spectrogram(Da_blk.T * (norm_val+1),'D_blk_NMF')
					###D_blk[:,:,ch] += Da_blk
				
					###Do IBM pre-processing (tone killing)
					##if IBM > 0:
					##	G_IBM_blk = X_blk[:,:,ch] > IBM*D_blk[:,:,ch]
					##	G_IBM_blk_tone = np.sum(G_IBM_blk,axis=0)
					##	G_IBM_blk_tone_idx = G_IBM_blk_tone > 0.25*param['val']['BLOCK_T']
					##	X_blk[:,G_IBM_blk_tone_idx,ch] = 0	

				##wavPlot.spectrogram(X_blk.T, None)

		#Reconstruct multichannel signals by unit mask
		G_um = np.maximum(G_mc[:,0],G_mc[:,1])
		for ch in range(ch_num):
			X_hat = Y_cpx[:,ch] * np.concatenate([G_um, np.flipud(G_um[1:MAG_SIZE-1])])
			#----GCtask3 options
			if CUT_FREQ > 0:
				cut_bin = int(CUT_FREQ/fs * MAG_SIZE)
				X_hat[:cut_bin] *= 1e-9
				X_hat[-cut_bin:] *= 1e-9

			x_hat_win[0,:,ch] = np.real(ifft(X_hat.T * (OLA_gain+0j), WINDOW_SIZE))
			x_hat_win[0,:,ch] = x_hat_win[0,:,ch] * OLA_win
			#----Reconstruct OLA
			x_hat_t = np.zeros(FRAME_SIZE)
			for i in range(OVERLAP_RATIO-1, -1, -1):
				x_hat_t = x_hat_t + x_hat_win[i, i * FRAME_SIZE : (i+1) * FRAME_SIZE,ch]
			x_hat_t = x_hat_t / (OVERLAP_RATIO / 2)
			if t_idx >= FRAME_SIZE * (OVERLAP_RATIO-1):
				x_hat[t_idx - FRAME_SIZE * (OVERLAP_RATIO-1):t_idx - FRAME_SIZE * (OVERLAP_RATIO-2),ch] = x_hat_t.astype('int16')

			#index update
			for i in range(OVERLAP_RATIO-1, 0, -1):
				x_hat_win[i, :,ch] = x_hat_win[i-1, :,ch]

		#Frame-wise VAD indices
		if param['vad']['is_vad'] == 1:		
			G_IBM = np.maximum(G_IBM_mc[:,0],G_IBM_mc[:,1])
			
			nb_harmonic_now = np.sum(G_IBM)
			nb_harmonic_delta_now = np.abs(nb_harmonic_now - nb_harmonic_now_1d)
			nb_harmonic.append(nb_harmonic_now)
			nb_harmonic_delta.append(nb_harmonic_delta_now)
			if nb_harmonic_now >= param['vad']['thr_harmonic'] and nb_harmonic_delta_now >= param['vad']['thr_harmonic_delta']:
				vad.append(1)
			else:
				vad.append(0)

			nb_harmonic_1d = nb_harmonic_now_1d

	x_hat = x_hat[pad.shape[0]:y_len_org,:] #Compensate block delay
	x_hat[:WINDOW_SIZE,:] = np.zeros((WINDOW_SIZE,ch_num))

	#VAD post-processing 
	frame_len_t = param['val']['FRAME_SIZE']/param['val']['FREQ_SAMPLE'] #16ms
	if param['vad']['is_vad'] == 1:		
		vad = np.array(vad)
		vad = vad[param['val']['BLOCK_T']-1:]
		vad[0] = 0
		#median filtering
		vad = vad_post_processing(vad, param['vad']['median_filter_coeff'])
		vad = np.array(vad,dtype=int)
		vad = np.array2string(vad, separator='')
		vad = vad.replace(' ','')
		vad = vad.replace('\n','')
		vad = vad[1:-1]
		min_speech_pattern = "1"*param['vad']['hang_cnt']
		if min_speech_pattern in vad:

			time_set = []
			#Find start, end pattern
			start_pattern = "01"
			end_pattern = "10"
			scan_pos = 0
			while scan_pos < len(vad):
				start_pos = vad[scan_pos:].find(start_pattern)+scan_pos
				end_pos = vad[scan_pos:].find(end_pattern)+scan_pos
				if start_pos >= end_pos:
					end_pos = vad[scan_pos+start_pos:].find(end_pattern) + start_pos
				
				if start_pos > int(param['vad']['start_sec']/frame_len_t) and end_pos > int(param['vad']['start_sec']/frame_len_t)+1:
					time_set.append([(start_pos+3-param['vad']['slack_len']) * frame_len_t, (end_pos+2+param['vad']['slack_len']) * frame_len_t])
					scan_pos += end_pos
				else:
					scan_pos += 1
					
			#merge consecutive VADs
			idx = 0
			for i in range(len(time_set)-1):
				if time_set[idx+1][0] - time_set[idx][1] < param['vad']['merge_thr']:
					time_set[idx][1] = time_set[idx+1][1]
					del time_set[idx+1]
					idx-=1
				idx+=1

			#cut too short VADs
			idx = 0
			for i in range(len(time_set)):
				if time_set[idx][1] - time_set[idx][0] < param['vad']['merge_thr']:
					del time_set[idx]
					idx-=1
				idx+=1

			nb_segment = len(time_set)
		else:
			nb_segment = 0
			time_set = []
		
		#Count number of speech segment
		if len(time_set) > 0:
			nb_segment = len(time_set)
		else:
			nb_segment = 0
			
		return x_hat, nb_segment, time_set

	else:
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
		[model, encoder, decoder] = nn_model.build_model_dae(FEAT_SHAPE, model_type, B_x, B_d, BATCH_SIZE, timestep)
		#with open(PATH_MODEL_X+'.json', 'w') as f:
		#	f.write(model.to_json())
		#with open(PATH_MODEL_X+'.enc.json', 'w') as f:
		#	f.write(encoder.to_json())
		#with open(PATH_MODEL_X+'.dec.json', 'w') as f:
		#	f.write(decoder.to_json())

		model.load_weights(PATH_MODEL_X)
		encoder.save_weights(PATH_MODEL_X+'.enc')
		decoder.save_weights(PATH_MODEL_X+'.dec') 
		encoder.load_weights(PATH_MODEL_X+'.enc')
		decoder.load_weights(PATH_MODEL_X+'.dec')
		models = model, encoder, decoder

	return models

def vad_post_processing(x, med_fil_len):
    med_fil_len_ = (med_fil_len - 1) // 2
    y = np.zeros((len(x), med_fil_len), dtype=x.dtype)
    y[:,med_fil_len_] = x
    for i in range (med_fil_len_):
        j = med_fil_len_ - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)