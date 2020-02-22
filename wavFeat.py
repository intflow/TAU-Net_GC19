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

PLOT = 0
SEP_RATIO = [0.8, 0.2] #Train, eval ratio
def feat_ext(event_name, DB_path, param):

	DB_read_seed = param['feat']['DB_read_seed']
	num_file_DB = param['feat']['num_file_DB']
	feat_type = param['feat']['feat_type']
	fs = param['feat']['fs']
	flen = param['feat']['flen']
	foverlap = param['feat']['foverlap']
	nFFT = param['feat']['nFFT']
	DC_Bin = param['feat']['DC_Bin']
	feat_len = param['feat']['feat_len']

	#Create folder
	event_path = "./feat/" + event_name +'/'+feat_type
	F.create_folder(event_path)
	F.create_folder(event_path+'/dev')
	F.create_folder(event_path+'/eval')

	#check whether retrain or not
	PATH_PICKLE = "./feat/" + event_name+"/data.pickle"
	TRAIN_FLAG = 1
	if os.path.isfile(PATH_PICKLE) :
		param_list_new = [event_name, DB_path, DB_read_seed, num_file_DB, feat_type, fs, flen, foverlap, nFFT, DC_Bin, feat_len]
		with open(PATH_PICKLE, 'rb') as f_event_info:
			param_list_old = pickle.load(f_event_info)

			if param_list_new == param_list_old :
				TRAIN_FLAG = 0
			else :
				TRAIN_FLAG = 1

	if TRAIN_FLAG == 1: #Train Begin

		#feature file path
		if not os.path.exists("./feat"):
			os.makedirs("feat")
		param_list = [event_name, DB_path, DB_read_seed, num_file_DB, feat_type, fs, flen, foverlap, nFFT, DC_Bin, feat_len]
		with open(PATH_PICKLE, 'wb') as f_event_info:
			pickle.dump(param_list, f_event_info)

		random.seed( DB_read_seed )

		#flist = [f for f in os.listdir(DB_path) if f.endswith('.wav')]
		flist = []
		for root, dirs, files in os.walk(DB_path, topdown=False):
			if not root[-3:] == 'bak':
				for name in files:
					flist = flist + [os.path.join(root, name)]
					#print(os.path.join(root, name))

		#Set number of files to be loaded
		if num_file_DB == 'whole':
			num_file_DB = len(flist)
			step = 1 #Load every data
		else:
			step = int(len(flist) / num_file_DB)
			if step < 1:
				step = 1       

		files = random.sample(flist, len(flist)) #Randomize file order
		num_file_tr = round(SEP_RATIO[0] * num_file_DB)
		num_file_eval = round(SEP_RATIO[1] * num_file_DB)

		#begin feature extraction
		file_cnt = 0
		for filename in files[::step]:
			ext = filename[-4-len(param['feat']['file_name_filter']):]
			
			if ext == param['feat']['file_name_filter']+'.wav':
				file_path = filename
				#file_path = file_path.replace("\\", "/")
				#---- Read wav files one by one ----
				#spf = wave.open(file_path, 'r')
				#(nchannels, sampwidth, framerate, nframes, comptype, compname) = spf.getparams()
				#dstr = spf.readframes(nframes * nchannels)
				#x = np.fromstring(dstr, numpy.int16)
				
				(x, fs_x) = sf.read(file_path, dtype='int16') 
				try:
					(_,ch_num) = x.shape
					if ch_num >= 2:
						x = x[:,0]
				except:
					x = x
				print("----Loading " + filename + ", fs: " + str(fs_x) + "----")
				#---- Resample wav files ----
				if fs_x != fs:
					x = resampy.resample(x * 0.5, sr_orig=fs_x, sr_new=fs)
					print("----Resample from " + str(fs_x) + "->" + str(fs) + "----")
				
				if PLOT == 1:
					wavPlot.waveform(x, None)

				#---- Apply filter effect on waveform
				#if filter_hpf == 1:
				#    x = F.highpass_filter(x, fs_x)

				#---- Get Feature for each wav clip ----
				if feat_type == 'STFT_pow' or feat_type == 'STFT_mag' or feat_type == 'STFT_logpow' or feat_type == 'STFT_mag_sqrt':
					(t, f, X) = sig.stft(x=x, fs=fs, window='hann', nperseg=flen, \
								noverlap=int(flen * foverlap), nfft=nFFT, detrend=False, return_onesided=True, \
								boundary='zeros', padded=True, axis=-1)
					X_mag = np.abs(X[:int(nFFT/2+1)]) #Magnitude of X
					print(X_mag.shape)
					if feat_type == 'STFT_mag':
						X_feat = X_mag
						print("----Extract STFT Magnitude Feature----")
					elif feat_type ==  'STFT_logpow':
						X_feat = np.log10(np.abs(X_mag)+0.0001) + 4 #Make power spectra to logscale
						print("----Extract log10 of STFT Magnitude Feature----")
					elif feat_type == 'STFT_mag_sqrt':
						X_feat = np.sqrt(X_mag)
						print("----Extract sqrt of STFT Magnitude Feature----")
					
					X_feat = np.transpose(X_feat)
					if PLOT == 1:
						wavPlot.spectrogram(X_feat, None)
				elif feat_type == 'mel_mag' or feat_type == 'mel_pow':
					X_feat=x
				elif feat_type == 'mfcc':
					X_feat=x
				elif feat_type == 'time_sample':
					pad_zero = np.zeros((int(np.ceil(len(x)/feat_len)*feat_len) - len(x),1))
					x = np.concatenate([np.reshape(x,(len(x),1)),pad_zero])
					X_feat = np.reshape(x,(int(len(x)/feat_len),feat_len))
					X_feat = X_feat.astype('int16')

				filename.replace("\\", "/")
				filename_name = filename.split('/')[-1][:-4]
				if file_cnt <= num_file_tr:
					PATH_FEAT_TRAIN = event_path + '/dev/'+filename_name+'.npy'
					f = open(PATH_FEAT_TRAIN, 'wb')
				else:
					PATH_FEAT_EVAL = event_path + '/eval/'+filename_name+'.npy'
					f = open(PATH_FEAT_EVAL, 'wb')
				
				np.save(f, X_feat)
			file_cnt = file_cnt + 1