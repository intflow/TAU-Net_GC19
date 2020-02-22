import numpy as np

def pesq2(reference, degraded, sample_rate=None, program='pesq'):
	import os
	""" Return PESQ quality estimation (two values: PESQ MOS and MOS LQO) based
	on reference and degraded speech samples comparison.
	Sample rate must be 8000 or 16000 (or can be defined reading reference file
	header).
	PESQ utility must be installed.
	"""
	if not os.path.isfile(reference) or not os.path.isfile(degraded):
		raise ValueError('reference or degraded file does not exist')
	if not sample_rate:
		import wave
		w = wave.open(reference, 'r')
		sample_rate = w.getframerate()
		w.close()
	if sample_rate not in (8000, 16000):
		raise ValueError('sample rate must be 8000 or 16000')
	import subprocess
	args = [ program, '+%d' % sample_rate, reference, degraded  ]
	pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
	out, _ = pipe.communicate()
	out = out.decode("utf-8")
	last_line = out.split('\r')[-2]
	chop = last_line.split('\t')[-2]
	score = chop.split(' ')[-1] #RAW PESQ score
	return score

#Original Author: Kamil Wojcicki, October 2011n MATLAB
def segSNR(x, x_hat, fs):
	from scipy import signal as sig
	# ensure masker is the same length as the x
	if len(x)!=len(x_hat):
		print( 'Error: length(x)~=length(y)')

	d = x_hat - x # compute the masker (assumes additive noise model)

	Tw = 32;                        # analysis frame duration (ms)
	flen = int(fs * Tw * 0.001)      
	nFFT = flen
	foverlap = 0.75

	ssnr_min = -1*10;                 # segment SNR floor (dB)
	ssnr_max =  35;                 # segment SNR ceil (dB)

	# divide x and masker signals into overlapped frames
	#frames.x = vec2frames( x, Nw, Ns, 'cols', @hanning, 0 );
	#frames.masker = vec2frames( masker, Nw, Ns, 'cols', @hanning, 0 );
	(f, t, X) = sig.stft(x=x, fs=fs, window='hann', nperseg=flen, \
						 noverlap=int(flen * foverlap), nfft=nFFT, detrend=False, return_onesided=True, \
						 boundary='zeros', padded=True, axis=-1)
	(f, t, D) = sig.stft(x=d, fs=fs, window='hann', nperseg=flen, \
						 noverlap=int(flen * foverlap), nfft=nFFT, detrend=False, return_onesided=True, \
						 boundary='zeros', padded=True, axis=-1)

	# compute x and masker frame energies
	#energy.x = sum( frames.x.^2, 1 );
	#energy.masker = sum( frames.masker.^2, 1 ) + eps;
	X_p = np.sum(np.abs(np.squeeze(X))**2, axis=0)
	D_p = np.sum(np.abs(np.squeeze(D))**2, axis=0)

	# compute frame signal-to-noise ratios (dB)
	ssnr = 10*np.log10(X_p / D_p + 0.0000000001)

	# apply limiting to segment SNRs
	ssnr = np.minimum(ssnr, ssnr_max)
	ssnr = np.maximum(ssnr, ssnr_min)

	# compute mean segmental SNR
	ssnr = np.mean(ssnr)
	return ssnr