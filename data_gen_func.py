import numpy as np
import keras
import random
from collections import deque
import os

class DataGenerator(object):
	'Generates data for Keras'
	def __init__(self, data_path_I='', data_path_O1='', data_path_O2='',
				 batch_size=16, seq_len=32, feat_len=512, n_channels=1, shuffle=True, per_file=False):
		'Initialization'
		self._per_file = per_file
		self._seq_len = seq_len
		self._feat_len = feat_len
		self._dim = (self._seq_len, self._feat_len)
		self._batch_size = batch_size
		self._batch_seq_len = self._batch_size*self._seq_len
		self._circ_buf_feat_i = deque()
		self._circ_buf_feat_o1 = deque()
		self._circ_buf_feat_o2 = deque()
		self._filenames_list = self.__get_file_list__(data_path_I)
		self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
		self._n_channels = n_channels
		self._data_path_I = data_path_I
		self._data_path_O1 = data_path_O1
		self._data_path_O2 = data_path_O2
		self._shuffle = shuffle
		self._nb_total_frames_ = self.__get_total_frames__()
		self.__on_epoch_end__()
		if self._per_file:
			self._nb_total_batches = len(self._filenames_list)
		else:
			self._nb_total_batches = int(np.floor((self._nb_total_frames_ / float(self._seq_len))))
		

	def __get_file_list__(self,path):
		import os
		'get file list from input folder'
		f_list = os.listdir(path)
		return f_list

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self._filenames_list) / self._batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self._batch_size:(index+1)*self._batch_size]

		# Find list of IDs
		_filenames_list_temp = [self._filenames_list[k] for k in indexes]

		# Generate data
		Y, X, D = self.generate()

		return Y, [X, D]

	def __on_epoch_end__(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self._filenames_list))
		if self._shuffle == True:
			np.random.shuffle(self.indexes)

	def __get_total_frames__(self):
		file_cnt = 0
		self._nb_total_frames_ = 0
		for file_cnt in range(len(self._filenames_list)):
			try:
				temp_feat_i = np.load(os.path.join(self._data_path_I, self._filenames_list[file_cnt]))
			except:
				print('erronous feature file checked')
				print(os.path.join(self._data_path_I, self._filenames_list[file_cnt]))
			self._nb_total_frames_ += temp_feat_i.shape[0]
		return self._nb_total_frames_

	def __feat_norm__(self, data):
		'Batch-wise normalization'
		nb_batch = data.shape[0]
		for b in range(nb_batch):
			data[b,] /= np.max(data[b,]) 
		return data

	def __feat_norm_group__(self, in1, out1, out2):
		'Batch-wise normalization by input denorm scale'
		nb_batch = in1.shape[0]
		for b in range(nb_batch):
			norm_scale = np.max(in1[b,:,:,:])
			in1[b,:,:,:] = in1[b,:,:,:] / norm_scale
			out1[b,:,:,:] = out1[b,:,:,:] / norm_scale
			out2[b,:,:,:] = out2[b,:,:,:] / norm_scale
		return in1, out1, out2

	def __split_in_seqs__(self, data):
		if len(data.shape) == 1:
			if data.shape[0] % self._seq_len:
				data = data[:-(data.shape[0] % self._seq_len), :]
			data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, 1))
		elif len(data.shape) == 2:
			if data.shape[0] % self._seq_len:
				data = data[:-(data.shape[0] % self._seq_len), :]
			data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1]))
		elif len(data.shape) == 3:
			if data.shape[0] % self._seq_len:
				data = data[:-(data.shape[0] % self._seq_len), :, :]
			data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1], data.shape[2]))
		else:
			print('ERROR: Unknown data dimensions: {}'.format(data.shape))
			exit()
		return data

	def get_data_sizes(self):
		feat_shape = (self._batch_size, self._seq_len, self._feat_len, self._n_channels)
		return feat_shape, [feat_shape, feat_shape]

	def get_total_batches(self):
		return self._nb_total_batches

	def generate(self, is_eval):
		'Generates data containing _batch_size samples' # X : (n_samples, *_dim, _n_channels)

		while 1:
		#if 1:
			if self._shuffle and is_eval == False:
				random.shuffle(self._filenames_list)
	
			# Ideally this should have been outside the while loop. But while generating the test data we want the data
			# to be the same exactly for all epoch's hence we keep it here.
			self._circ_buf_feat_i = deque()
			self._circ_buf_feat_o1 = deque()
			self._circ_buf_feat_o2 = deque()
	
			file_cnt = 0
			##for i in range(len(self._filenames_list)):
			# load feat and label to circular buffer. Always maintain at least one batch worth feat and label in the
			# circular buffer. If not keep refilling it.
			buff_cnt = 0
			#print(buff_cnt)
			while buff_cnt < self._batch_seq_len:
				temp_feat_i = np.load(os.path.join(self._data_path_I, self._filenames_list[file_cnt]))[:,1:]
				temp_feat_o1 = np.load(os.path.join(self._data_path_O1, self._filenames_list[file_cnt]))[:,1:]
				temp_feat_o2 = np.load(os.path.join(self._data_path_O2, self._filenames_list[file_cnt]))[:,1:]

				if np.sum(np.isnan(temp_feat_i) + np.isnan(temp_feat_o1) + np.isnan(temp_feat_o2)):
					print('!!!!!NaN Detected!!!!!!')
				else:
					for row_cnt, row in enumerate(temp_feat_i):
						self._circ_buf_feat_i.append(row)
						self._circ_buf_feat_o1.append(temp_feat_o1[row_cnt])
						self._circ_buf_feat_o2.append(temp_feat_o2[row_cnt])
						buff_cnt += 1
						
					# If self._per_file is True, this returns the sequences belonging to a single audio recording
					if self._per_file:
						extra_frames_i = self._batch_seq_len - temp_feat_i.shape[0]
						extra_frames_o1 = self._batch_seq_len - temp_feat_o1.shape[0]
						extra_frames_o2 = self._batch_seq_len - temp_feat_o2.shape[0]
						extra_feat_i = np.ones((extra_frames_i, temp_feat_i.shape[1])) * 1e-6
						extra_feat_o1 = np.ones((extra_frames_o1, temp_feat_o1.shape[1])) * 1e-6
						extra_feat_o2 = np.ones((extra_frames_o2, temp_feat_o2.shape[1])) * 1e-6

						for row_cnt, row in enumerate(extra_feat_i):
							self._circ_buf_feat_i.append(row)
							self._circ_buf_feat_o1.append(extra_feat_o1[row_cnt])
							self._circ_buf_feat_o2.append(extra_feat_o2[row_cnt])
							buff_cnt += 1

				file_cnt = file_cnt + 1

				#Reshuffle if the file is insufficient to make one batch
				if len(self._filenames_list) == file_cnt:
					file_cnt = 0
			#print(buff_cnt)

			# Read one batch size from the circular buffer
			feat_i = np.ones((self._batch_seq_len, self._feat_len * self._n_channels)) * 1e-6
			feat_o1 = np.ones((self._batch_seq_len, self._feat_len * self._n_channels)) * 1e-6
			feat_o2 = np.ones((self._batch_seq_len, self._feat_len * self._n_channels)) * 1e-6

			try:
				for j in range(self._batch_seq_len):
					feat_i[j, :] = self._circ_buf_feat_i.popleft()
					feat_o1[j, :] = self._circ_buf_feat_o1.popleft()
					feat_o2[j, :] = self._circ_buf_feat_o2.popleft()
			except:
				print('Buffer Error Detected')

			feat_i = np.reshape(feat_i, (self._batch_seq_len, self._feat_len, self._n_channels))
			feat_o1 = np.reshape(feat_o1, (self._batch_seq_len, self._feat_len, self._n_channels))
			feat_o2 = np.reshape(feat_o2, (self._batch_seq_len, self._feat_len, self._n_channels))

			# Split to sequences
			feat_i = self.__split_in_seqs__(feat_i)
			feat_o1 = self.__split_in_seqs__(feat_o1)
			feat_o2 = self.__split_in_seqs__(feat_o2)

			# Data normalization (Max norm per batch)
			feat_i, feat_o1, feat_o2 = self.__feat_norm_group__(feat_i, feat_o1, feat_o2)
			
			####DEBUG
			#import wavPlot
			#wavPlot.contour(np.reshape(feat_i[0,:,:,:],(self._seq_len,self._feat_len)).T,None)
			#wavPlot.contour(np.reshape(feat_o1[0,:,:,:]+feat_o2[0,:,:,:],(self._seq_len,-1)).T,None)
			#wavPlot.contour(np.reshape(feat_o1[0,:,:,:],(self._seq_len,-1)).T,None)
			#wavPlot.contour(np.reshape(feat_o2[0,:,:,:],(self._seq_len,-1)).T,None)
			
			yield feat_i, [feat_o1, feat_o2, feat_o1, feat_o2]
