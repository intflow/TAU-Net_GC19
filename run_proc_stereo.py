import wavProc
import wavProc_UM
import os
import soundfile as sf
import argparse
import numpy as np
from eval_utils import bss_eval
import json
from eval_utils.eval_subfunc import pesq2, segSNR
#from pystoi.stoi import stoi
import argparse
import resampy
import scipy
import wavPlot
#import pyloudnorm as pyln
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from subfuncs import create_folder

def main(param, args):


	#--predefined parameters
	CLEAN_PATH = param['path']['CLEAN_PATH']
	NOISY_PATH = param['path']['NOISY_PATH']
	PROC_PATH = param['path']['PROC_PATH']
	PATH_MODEL_X = param['path']['PATH_MODEL_X']
	PATH_MODEL_D = param['path']['PATH_MODEL_D']
	PATH_SUBMODEL_X = param['path']['PATH_SUBMODEL_X']
	PATH_SUBMODEL_D = param['path']['PATH_SUBMODEL_D']
	PROC_METHOD = param['path']['PROC_METHOD']
	CALC_TAG = param['path']['CALC_TAG']
	ALIAS = param['path']['ALIAS']
	UNIT_MASK = param['val']['UNIT_MASK']
	LOUDNESS_NORM = param['val']['LOUDNESS_NORM']
	DB_SPLIT = param['val']['DB_SPLIT'] #"1/3", "2/3", "3/3"
	GPU_IDX = param['val']['GPU_IDX'] #"0","1","2","3'...

	#Multi-GPU setting
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.visible_device_list = GPU_IDX
	set_session(tf.Session(config=config))


	#----Make folder for processed outputs----
	create_folder(PROC_PATH)
	if param['vad']['is_vad'] == 1:
		LABEL_PATH = PROC_PATH+'_TAUvad_label'
		create_folder(LABEL_PATH)

	SAVE_PROC = args["savefile"]
	BYPASS = args["bypass"]
	TASK_NAME = args["task"]

	#--Initial buffers
	sdr_total = np.empty((1,))
	sir_total = np.empty((1,))
	sar_total = np.empty((1,))
	pesq_total = np.empty((1,))
	stoi_total = np.empty((1,))
	ssnr_total = np.empty((1,))

	FREQ_SAMPLE = param['val']['FREQ_SAMPLE']
	FRAME_SIZE = param['val']['FRAME_SIZE']
	OVERLAP_RATIO = param['val']['OVERLAP_RATIO']
	BLOCK_T = param['val']['BLOCK_T']
	WINDOW_SIZE = FRAME_SIZE * OVERLAP_RATIO
	MAG_SIZE = int(WINDOW_SIZE / 2 + 1)

	if BYPASS == 0:
		#--load nnet model
		if PROC_METHOD == 'TAU-Net':
			models = wavProc.load_sep_model(MAG_SIZE, BLOCK_T, PROC_METHOD, PATH_MODEL_X, PATH_MODEL_D, PATH_SUBMODEL_X, PATH_SUBMODEL_D)

	#----Load Clean and Noisy Audio File
	IS_PATH=0
	for (path, dir, files) in os.walk(CLEAN_PATH):
		IS_PATH=1
		#Set number of files to be loaded
		if DB_SPLIT == 'whole':
			DB_SPLIT = len(files)
			files = files[:DB_SPLIT]
		elif DB_SPLIT == '1/4':
			DB_SPLIT = int(0.25*len(files))
			files = files[:DB_SPLIT]
		elif DB_SPLIT == '2/4':
			DB_SPLIT = int(0.25*len(files))
			files = files[DB_SPLIT:2*DB_SPLIT]
		elif DB_SPLIT == '3/4':
			DB_SPLIT = int(0.25*len(files))
			files = files[2*DB_SPLIT:3*DB_SPLIT]
		elif DB_SPLIT == '4/4':
			DB_SPLIT = int(0.25*len(files))
			files = files[3*DB_SPLIT:]


		for filename in files:
			ext = os.path.splitext(filename)[-1]

			if ext == '.wav' and (filename.find(CALC_TAG)) == 0:
				#Consider whole file set is matched to CLEAN_PATH
				x_path = CLEAN_PATH + "/" + filename
				y_path = NOISY_PATH + "/" + filename
				x_hat_path = PROC_PATH + "/" + filename
				(x_2ch, fs_x) = sf.read(x_path, dtype='int16')
				(y_2ch, fs_y) = sf.read(y_path, dtype='int16')
				#print("----Loading " + filename + ", fs: " + str(fs_x) + "----")

				[T,CH] = x_2ch.shape

				if UNIT_MASK == 1:
					#---- Resample wav files ----
					if fs_x != FREQ_SAMPLE:
						x_2ch = resampy.resample(x_2ch.T * 0.5, sr_orig=fs_x, sr_new=FREQ_SAMPLE)
						#print("----Resample from " + str(fs_x) + "->" + str(FREQ_SAMPLE) + "----")
					if fs_y != FREQ_SAMPLE:
						y_2ch = resampy.resample(y_2ch.T * 0.5, sr_orig=fs_y, sr_new=FREQ_SAMPLE)
						#print("----Resample from " + str(fs_y) + "->" + str(FREQ_SAMPLE) + "----")

					if PROC_METHOD == 'TAU-Net':
						if param['vad']['is_vad'] == 1:	
							x_hat_2ch, nb_segment, time_set = wavProc_UM.TAU_Net(y_2ch.T, FREQ_SAMPLE, models, param, filename, PROC_METHOD, PATH_SUBMODEL_X, PATH_SUBMODEL_D)
							f_vad = open(LABEL_PATH+'/'+filename[:-4]+'.txt', 'w')
							
							str_vad = ['#segments	'+str(nb_segment)]
							for time_set_sub in time_set:
								str_time = "{:.2f}".format(time_set_sub[0])+'	'+"{:.2f}".format(time_set_sub[1])
								str_vad.append(str_time)
							f_vad.writelines('\n'.join(str_vad))
							
							str_tmp = ' '.join(str_vad)
							str_tmp.replace('\t',' ')
							print(filename+ ': '+ str_tmp)
						else:
							x_hat_2ch = wavProc_UM.TAU_Net(y_2ch.T, FREQ_SAMPLE, models, param, filename, PROC_METHOD, PATH_SUBMODEL_X, PATH_SUBMODEL_D)

				else:
					for ch in range(0,CH):
						y = y_2ch[:,ch]
						x = x_2ch[:,ch]
											
						#---- Resample wav files ----
						if fs_x != FREQ_SAMPLE:
							x = resampy.resample(x * 0.5, sr_orig=fs_x, sr_new=FREQ_SAMPLE)
							print("----Resample from " + str(fs_x) + "->" + str(FREQ_SAMPLE) + "----")
						if fs_y != FREQ_SAMPLE:
							y = resampy.resample(y * 0.5, sr_orig=fs_y, sr_new=FREQ_SAMPLE)
							print("----Resample from " + str(fs_y) + "->" + str(FREQ_SAMPLE) + "----")
						
						if BYPASS == 0:
							##----Begin process by defined method
							if PROC_METHOD == 'TAU-Net':
								x_hat = wavProc.TAU_Net(y, FREQ_SAMPLE, models, param, filename, PROC_METHOD, PATH_SUBMODEL_X, PATH_SUBMODEL_D)

							if len(y) > len(x_hat):
								len_x = len(x_hat)
							else:
								len_x = len(y)
							y = np.reshape(y[:len_x],(1, len_x))
							x = np.reshape(x[:len_x],(1, len_x))
							x_hat = np.reshape(x_hat,(1, len_x))

							#----Evaluate target and estimated files
							#try:
							#	 pesq = pesq2(x_path, x_hat_path, sample_rate=16000, program='PESQ2.exe')
							#	 (sdr, sir, sar, _) = bss_eval.bss_eval_sources(np.concatenate([x, y-x+1]), np.concatenate([x_hat, y-x_hat]), False) #score = [sdr, sir, sar, popt]
							#	 stoi_val = stoi(np.squeeze(x), np.squeeze(x_hat), fs_x, extended=False)
							#	 ssnr = segSNR(x, x_hat, fs_x)
							#except:
							#	 pesq = 1.0
							#	 sdr, sir, sar = [[0.0], [0.0], [0.0]]
							#	 stoi_val = 0.0
							#	 ssnr = 0.0
							
							if ch == 0:
								x_hat_2ch = x_hat.T
							else:
								x_hat_2ch = np.concatenate([x_hat_2ch,x_hat.T],axis=-1)

						#---Use only for already processed files---
						elif BYPASS == 1:
							(x_hat_2ch, fs_x) = sf.read(x_hat_path, dtype='int16')

					###----Mid Side back to L/R
					##x_hat_2ch_buff = x_hat_2ch
					##x_hat_2ch[:,0] = (x_hat_2ch_buff[:,0] + x_hat_2ch_buff[:,1]) * 2.0
					##x_hat_2ch[:,1] = (x_hat_2ch_buff[:,0] - x_hat_2ch_buff[:,1]) * 2.0
				x_hat_2ch = x_hat_2ch * 2.0
				if LOUDNESS_NORM == 1:
					# peak normalize audio to -1 dB
					peak_normalized_audio = pyln.normalize.peak(x_hat_2ch, -1.0)

					# measure the loudness first 
					meter = pyln.Meter(FREQ_SAMPLE) # create BS.1770 meter
					loudness = meter.integrated_loudness(x_hat_2ch)

					# loudness normalize audio to -12 dB LUFS
					x_hat_2ch = pyln.normalize.loudness(x_hat_2ch, loudness, -20.0)
				else:
					x_hat_2ch /= 32768.0
				if SAVE_PROC == 1:
					sf.write(x_hat_path, x_hat_2ch, FREQ_SAMPLE, 'PCM_16')

					#sdr_total = np.append(sdr_total, sdr[0])
					#sir_total = np.append(sir_total, sir[0])
					#sar_total = np.append(sar_total, sar[0])
					###pesq_total = np.append(pesq_total, float(pesq))
					#stoi_total = np.append(stoi_total, stoi_val)
					#ssnr_total = np.append(ssnr_total, float(ssnr))

	if IS_PATH == 0:
		print("--------------------------------------------")
		print("----Check File Path Vaild!!!, Exiting...----")
		print("--------------------------------------------")
		return 0

	##----final result
	#sdr_mu = round(np.mean(sdr_total[1:]), 4)
	#sdr_var = round(np.sqrt(np.var(sdr_total[1:])), 4)
	#sir_mu = round(np.mean(sir_total[1:]), 4)
	#sir_var = round(np.sqrt(np.var(sir_total[1:])), 4)
	#sar_mu = round(np.mean(sar_total[1:]), 4)
	#sar_var = round(np.sqrt(np.var(sar_total[1:])), 4)
	#pesq_mu = round(np.mean(pesq_total[1:]), 4)
	#pesq_var = round(np.sqrt(np.var(pesq_total[1:])), 4)
	#stoi_mu = round(np.mean(stoi_total[1:]), 4)
	#stoi_var = round(np.sqrt(np.var(stoi_total[1:])), 4)
	#ssnr_mu = round(np.mean(ssnr_total[1:]), 4)
	#ssnr_var = round(np.sqrt(np.var(ssnr_total[1:])), 4)
#
	#result_print = 'SDR({}, {}) SIR({}, {}) SAR({}, {}) PESQ({}, {}) STOI({}, {}) SSNR({}, {})'.format(
	#		sdr_mu, sdr_var, sir_mu, sir_var, sar_mu, sar_var, pesq_mu, pesq_var, stoi_mu, stoi_var, ssnr_mu, ssnr_var)
	#print(result_print)
#
	#result_txt = '{}, {}\n{}, {}\n{}, {}\n{}, {}\n{}, {}\n{}, {}'.format(
	#		sdr_mu, sdr_var, sir_mu, sir_var, sar_mu, sar_var, pesq_mu, pesq_var, stoi_mu, stoi_var, ssnr_mu, ssnr_var)
	#
	#RESULT_PATH = 'results/'+TASK_NAME+'/'
	#if (not os.path.exists(RESULT_PATH)):
	#	 os.makedirs(RESULT_PATH)
	#f = open(RESULT_PATH + ALIAS + CALC_TAG+'.txt','wt')
	#f.write(result_txt)


if __name__ == '__main__':
	
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--configs", type=str, default="configs_proc/TAUnet_GCtask3_stereo.json",
		help="load json config file")
	ap.add_argument("-t", "--task", type=str, default="",
		help="Task name (result will be saved in a folder of this name)")
	ap.add_argument("-b", "--bypass", type=int, default=0,
		help="bypass processing jump into score check")
	ap.add_argument("-f", "--savefile", type=int, default=1,
		help="check save file or not")
	ap.add_argument("-di", "--dir", type=str, default="None",
		help="check save file or not")
	ap.add_argument("-do", "--out_dir", type=str, default="None",
		help="check save file or not")
	ap.add_argument("-gpuidx", "--gpu_idx", type=str, default="0",
		help="check save file or not")
	ap.add_argument("-dbs", "--db_split", type=str, default="whole",
		help="check save file or not")
	args = vars(ap.parse_args())

	json_param = open(args["configs"]).read()
	param = json.loads(json_param)

	
	if args["dir"] != "None":
		param['path']['CLEAN_PATH'] = args["dir"]
		param['path']['NOISY_PATH'] = args["dir"]
	if args["out_dir"] != "None":
		param['path']['PROC_PATH'] = args["out_dir"]
	
	param['val']['GPU_IDX'] = args["gpu_idx"]
	param['val']['DB_SPLIT'] = args["db_split"]
		
	if param['path']['CALC_TAG'] == "None":
		param['path']['CALC_TAG'] = ""
		main(param, args)
	else :
		SNR_iter = ['_0_', '_5_', '_10_', '_15_']
		for SNR in SNR_iter:
			param['path']['CALC_TAG'] = SNR
			main(param, args)
