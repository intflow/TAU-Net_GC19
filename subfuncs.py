import numpy as np
from scipy import signal
import os

#----Sub-functions----
def overlap_spec_3D(X, pad, step, even_odd, norm, norm_given=None):
    #Consider X.shape = [t, f]
    #Pad zeros to edge frames
    [i, f] = X.shape
    Xp = np.ones((pad,f)) * 0.01
    if even_odd == 1: #Oddoverlap_spec_3D
        X = np.concatenate((Xp, X), axis=0)
        X = np.concatenate((X, Xp), axis=0)
        t = 2*pad + 1
        pad_e = pad + 1
    else: #even
        X = np.concatenate((Xp, X), axis=0)
        X = np.concatenate((X, Xp), axis=0)
        t = 2*pad
        pad_e = pad

    #Stack pilar of spectrogram
    i = int(i/step)
    X_3D = np.ones((i, t, f))
    scales = np.ones((i, 1))
    for it in range(0, i):
        i_c = step*it + pad
        
        if norm == 'L1':
            try:
                if norm_given == None:
                    scales[it,:] = np.sum(np.sum(X[i_c - pad : i_c + pad_e:, :]))
            except:
                scales[it,:] = norm_given[it,:]

        if norm == 'L2':
            try:
                if norm_given == None:
                    scales[it,:] = np.sqrt(np.sum(np.sum(np.power(X[i_c - pad : i_c + pad_e:, :],2))))
            except:
                scales[it,:] = norm_given[it,:]

        if norm == 'MAX':
            try:
                if norm_given == None:
                    scales[it,:] = np.amax(X[i_c - pad : i_c + pad_e:, :])
            except:
                scales[it,:] = norm_given[it,:]

        if norm == 'LogMAX':
            X[i_c - pad : i_c + pad_e:, :] = np.log10(np.square(X[i_c - pad : i_c + pad_e:, :]) + 1.0001)
            try:
                if norm_given == None:
                    scales[it,:] = np.amax(X[i_c - pad : i_c + pad_e:, :])
            except:
                scales[it,:] = norm_given[it,:]

        X_3D[it,:,:] = X[i_c - pad : i_c + pad_e:, :] / (scales[it,:] + 1)
        
    return [X_3D, scales]        

def get_IBM(X, N, LC):
    LC_MAP = 10*(np.log10(X+0.001) - np.log10(N+0.001))
    M = np.greater(LC_MAP, LC).astype(int)
    return M

def highpass_filter(y, sr):

    filter_stop_freq = 800  # Hz
    filter_pass_freq = 1000  # Hz
    filter_order = 2 

    # High-pass filter
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate) 
    
    # Apply high-pass filter
    filtered_audio = signal.filtfilt(filter_coefs, [1], y)
    return filtered_audio

def denorm_logspec(x):
    BIAS_VAL = 0.01831563890110627016310263418666 #pow(exp,-4)
    FLOOR = 8.0001
    x_denorm = np.exp((x - FLOOR) * 0.5) - BIAS_VAL
    return x_denorm	
	
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)
		
