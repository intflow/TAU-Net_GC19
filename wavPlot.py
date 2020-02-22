import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def waveform(x, pic_path):
    if pic_path == None:
        pic_path = ".wav.tmp"
    plt.figure()
    plt.plot(x)
    plt.savefig(pic_path + ".png", bbox_inches='tight')
    plt.show()

def spectrogram (X, pic_path): 
    if pic_path == None:
        pic_path = "spec.tmp."
    logX_feat = np.log10(X+0.001)
    zmin = logX_feat.min()
    zmax = logX_feat.max()
    plt.figure()
    plt.contourf(logX_feat, cmap='inferno', vmin=zmin, vmax=zmax)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    
    if pic_path == None:
        plt.show()
    else:
        try:
            plt.savefig(pic_path + ".png", bbox_inches='tight')
        except:
            return 0

def spectrogram_batch (X, pic_path): 
    X = np.squeeze(X)
    b,t,f = X.shape
    if b > 32:
        b = 32
        X = X[:b,:,:]
    X = np.reshape(X,(b*t,f))
    X = X.T
    if pic_path == None:
        pic_path = "spec.tmp."
    logX_feat = np.log10(X+0.001)
    zmin = logX_feat.min()
    zmax = logX_feat.max()
    plt.figure()
    plt.contourf(logX_feat, cmap='inferno', vmin=zmin, vmax=zmax)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    
    if pic_path == None:
        plt.show()
    else:
        try:
            plt.savefig(pic_path + ".png", bbox_inches='tight')
        except:
            return 0            

def contour (X, pic_path): 
    if pic_path == None:
        pic_path = "cont.tmp."
    zmin = X.min()
    zmax = X.max()
    plt.figure()
    plt.contourf(X, cmap='inferno', vmin=zmin, vmax=zmax)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Rank')
    plt.colorbar()
    plt.savefig(pic_path + ".png", bbox_inches='tight')
    plt.show()