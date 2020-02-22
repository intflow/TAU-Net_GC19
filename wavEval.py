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

from tensorflow import keras
from keras.callbacks import History, ModelCheckpoint
from sparse_nmf import sparse_nmf
from sklearn.cluster import KMeans


CPU_MODE = 0
BATCH_SIZE_TEST = 1
def eval_model(event_name, model_type):
    if CPU_MODE == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

    #Load feature info
    PATH_PICKLE = "./feat/" + event_name + ".feat.pickle"
    with open(PATH_PICKLE, 'rb') as f_event_info:
        [event_name, DB_path, DB_read_seed, num_file_DB, feat_type, \
        fs, flen, foverlap, nFFT, DC_Bin, feat_len] = pickle.load(f_event_info)

    FEAT_SHAPE = (1,feat_len)
    PATH_MODEL_CHECKPOINT = "./model/" + event_name + "." + model_type + ".best.model"
    
    
    #----Define model structure then compile-----
    [model, encoder, decoder] = nn_model.build_model_ae(FEAT_SHAPE, feat_len, model_type)

    #Load previous model and history
    model.load_weights(PATH_MODEL_CHECKPOINT)
    encoder.load_weights(PATH_MODEL_CHECKPOINT)
    decoder.load_weights(PATH_MODEL_CHECKPOINT)

    #----Extract specific weigths----
    weights_list = model.get_weights() 

    #----Reconstruct Data from Model (Evaluation)----
    #load features (Eval)
    PATH_FEAT_EVAL = "./feat/" + event_name + ".feat.eval"
    f_event_feat_eval = open(PATH_FEAT_EVAL, 'rb')
    X_eval = np.load(f_event_feat_eval)
    [t, f] = X_eval.shape
    X_eval = np.reshape(X_eval, (t, 1, f))
    print("Finished loading evaluation dataset")

    #Conduct inference
    X_hat = model.predict(X_eval, batch_size=BATCH_SIZE_TEST)
    
    #Evaluate results
    [score, mae] = model.evaluate(X_eval, X_hat, batch_size=BATCH_SIZE_TEST)
    rmse = np.sqrt(((X_hat - X_eval) ** 2).mean(axis=0)).mean()
    
    print('RMSE:' + str(rmse))   # Printing RMSE
    print('Score:' + str(score))
    print('MAE:' + str(mae))
    
    #Save objective scores to txt
    PATH_RESULT = "./output/" + event_name + "." + model_type + ".result.txt"

    with open(PATH_RESULT, 'w') as fout:
        fout.write('RMSE:' + str(rmse))
        fout.write('\nScore:' + str(score))
        fout.write('\nMAE:' + str(mae))

    #Generate some random outputs from Decoder
    if model_type == 'VAE':
        #z_gen = np.random.rand(t, 10)
        enc_out = encoder.predict(X_eval, batch_size=BATCH_SIZE_TEST)
        rand_rate = 0.3       
        z_gen = (1-rand_rate) * enc_out[2] + rand_rate * np.random.rand(t, 1, 10)
        z_gen = np.squeeze(z_gen)
        X_gen = decoder.predict(z_gen, batch_size=BATCH_SIZE_TEST)

    #Show Sample Spectrogram as evaluation
    PATH_PIC = "./output/" + event_name + "." + model_type
    X_eval = np.squeeze(X_eval)
    wavPlot.spectrogram(np.exp(X_eval[0:100,:]), PATH_PIC + ".X.sample")
    X_hat = np.squeeze(X_hat)
    wavPlot.spectrogram(np.exp(X_hat[0:100,:]), PATH_PIC + ".X_hat.sample")
    
    if model_type == 'VAE':
        wavPlot.spectrogram(np.exp(X_gen[0:100,:]), PATH_PIC + ".X_gen.sample") 


def eval_model_NMF(event_name, model_type, param):

    max_iter=param['nmf']['max_iter']
    random_seed=param['nmf']['random_seed']
    sparsity=param['nmf']['sparsity']
    conv_eps=param['nmf']['conv_eps']
    cf=param['nmf']['cf']
    display=param['nmf']['display']
    cost_check=param['nmf']['cost_check']

    #Load feature info
    PATH_PICKLE = "./feat/" + event_name + ".feat.pickle"
    with open(PATH_PICKLE, 'rb') as f_event_info:
        [event_name, DB_path, DB_read_seed, num_file_DB, feat_type, \
        fs, flen, foverlap, nFFT, DC_Bin, feat_len] = pickle.load(f_event_info)

    PATH_MODEL = "./model_NMF/" + event_name + "." + model_type
    
    #with open(PATH_MODEL, 'rb') as f_basis:
    #    w = np.load(f_basis)
    #with open(PATH_MODEL_ACTIVATION, 'rb') as f_activation:
    #    h = np.load(f_activation)
    w = np.loadtxt(PATH_MODEL + ".basis")
    #h = np.loadtxt(PATH_MODEL + ".activation")

    (k,r)=w.shape

    #Check Basis by Images
    wavPlot.contour(w, PATH_MODEL + "_R{}.w".format(r))


    #----Reconstruct Data from Model (Evaluation)----
    #load features (Eval)
    PATH_FEAT_EVAL = "./feat/" + event_name + ".feat.eval"
    f_event_feat_eval = open(PATH_FEAT_EVAL, 'rb')
    X_eval = np.load(f_event_feat_eval)
    X_eval = X_eval.T

    (f, t) = X_eval.shape
    if t > 200:
        X_eval = X_eval[:,600:700]
    print("Finished loading evaluation dataset")

    #Conduct inference (Supervised NMF)   
    W_IND = range(0,0)
    H_IND = range(0,r)
    init_w = w #pretrained basis from training set
    [w_eval, h_eval, obj] = sparse_nmf(X_eval, max_iter, random_seed, sparsity, conv_eps, cf, \
                                init_w=init_w, init_h=None, w_ind=W_IND, h_ind=H_IND, r=r, display=display, cost_check=cost_check)
    #Reconstruct spectra
    X_hat = w @ h_eval
    
    #Evaluate results
    rmse = np.sqrt(((X_hat - X_eval) ** 2).mean(axis=0)).mean()
    
    print('RMSE:' + str(rmse))   # Printing RMSE
    
    #Save objective scores to txt
    PATH_RESULT = "./output/" + event_name + "." + model_type + ".result.txt"

    with open(PATH_RESULT, 'w') as fout:
        fout.write('RMSE:' + str(rmse))

    #Show Sample Spectrogram as evaluation
    PATH_PIC = "./output/" + event_name + "." + model_type

    if feat_type == 'STFT_logpow' :
        X_eval = np.power(10, (X_eval - 8.001)*0.5)-1e-4
        X_hat = np.power(10, (X_hat - 8.001)*0.5)-1e-4

    if feat_type == 'STFT_mag_sqrt':
        X_eval = np.square(X_eval)
        X_hat = np.square(X_hat)

    wavPlot.spectrogram(X_eval[:,0:100], PATH_PIC + ".X.sample")
    wavPlot.spectrogram(X_hat[:,0:100], PATH_PIC + ".X_hat.sample")


def eval_model_discrim_sep(input_name, target_name, noise_name, model_type):
    if CPU_MODE == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

    #Load feature info
    PATH_PICKLE = "./feat/" + target_name + ".feat.pickle"
    with open(PATH_PICKLE, 'rb') as f_event_info:
        [input_name, DB_path, DB_read_seed, num_file_DB, feat_type, \
        fs, flen, foverlap, nFFT, DC_Bin, feat_len] = pickle.load(f_event_info)

    PATH_MODEL_CHECKPOINT = "./model/" + input_name + "." + model_type + ".best.model"
    PATH_MODEL_CHECKPOINT_ENC = "./model/" + input_name + "." + model_type + ".best.model.enc"
    PATH_MODEL_CHECKPOINT_DEC = "./model/" + input_name + "." + model_type + ".best.model.dec"
    

    #----Reconstruct Data from Model (Evaluation)----
    if model_type == 'DRNN':

        FEAT_SHAPE = (1,feat_len)

        #----Define model structure then compile-----
        model = nn_model.build_model_discrim_sep(FEAT_SHAPE, feat_len, model_type)
        #Load previous model and history
        model.load_weights(PATH_MODEL_CHECKPOINT)
        #----Extract specific weigths----
        weights_list = model.get_weights() 

        #load features (Eval)
        PATH_FEAT_EVAL_I = "./feat/" + input_name + ".feat.eval"
        PATH_FEAT_EVAL_O1 = "./feat/" + target_name + ".feat.eval"
        PATH_FEAT_EVAL_O2 = "./feat/" + noise_name + ".feat.eval"
        f_event_feat_eval_I = open(PATH_FEAT_EVAL_I, 'rb')
        f_event_feat_eval_O1 = open(PATH_FEAT_EVAL_O1, 'rb')
        f_event_feat_eval_O2 = open(PATH_FEAT_EVAL_O2, 'rb')
        Y = np.load(f_event_feat_eval_I)
        X = np.load(f_event_feat_eval_O1)
        D = np.load(f_event_feat_eval_O2)
        [t, f] = Y.shape
        Y = np.reshape(Y, (t, 1, f))
        X = np.reshape(X, (t, 1, f))
        D = np.reshape(D, (t, 1, f))
        Y = np.log(Y+1)
        X = np.log(X+1)
        D = np.log(D+1)
        print("Finished loading evaluation dataset")

        #Conduct inference
        [X_hat, D_hat, Y_hat] = model.predict(Y, batch_size=BATCH_SIZE_TEST)
        X_hat = np.exp(X_hat) - 1
        D_hat = np.exp(D_hat) - 1
        
        #Evaluate results
        #[score_x, mae_x] = model.evaluate(X, X_hat, batch_size=BATCH_SIZE_TEST)
        #[score_d, mae_d] = model.evaluate(D, D_hat, batch_size=BATCH_SIZE_TEST)
        rmse_x = np.sqrt(((X_hat - X) ** 2).mean(axis=0)).mean()
        rmse_d = np.sqrt(((D_hat - D) ** 2).mean(axis=0)).mean()
        
        #print('RMSE_x:' + str(rmse_x))   # Printing RMSE
        #print('Score_x:' + str(score_x))
        #print('MAE_x:' + str(mae_x))
        #print('RMSE_d:' + str(rmse_d))   # Printing RMSE
        #print('Score_d:' + str(score_d))
        #print('MAE_d:' + str(mae_d))
    #
        #Save objective scores to txt
        PATH_RESULT = "./output/" + input_name + "." + model_type + ".result.txt"
    #
        with open(PATH_RESULT, 'w') as fout:
            fout.write('RMSE_x:' + str(rmse_x))
            #fout.write('\nScore_x:' + str(score_x))
            #fout.write('\nMAE_x:' + str(mae_x))
            fout.write('RMSE_d:' + str(rmse_d))
            #fout.write('\nScore_d:' + str(score_d))
            #fout.write('\nMAE_d:' + str(mae_d))

        #Show Sample Spectrogram as evaluation
        PATH_PIC = "./output/" + input_name + "." + model_type
        Y = np.squeeze(Y)
        wavPlot.spectrogram(np.exp(Y[0:100,:]), PATH_PIC + ".Y.sample")
        #X = np.squeeze(X)
        #wavPlot.spectrogram(X[0:100,:], PATH_PIC + ".X.sample")
        X_hat = np.squeeze(X_hat)
        wavPlot.spectrogram(np.exp(X_hat[0:100,:]), PATH_PIC + ".X_hat.sample")
        #D = np.squeeze(D)
        #wavPlot.spectrogram(D[0:100,:], PATH_PIC + ".D.sample")
        D_hat = np.squeeze(D_hat)
        wavPlot.spectrogram(np.exp(D_hat[0:100,:]), PATH_PIC + ".D_hat.sample")

    if model_type == 'U-Net':

        FEAT_SHAPE = (128, feat_len-1, 1)

        #----Define model structure then compile-----
        model = nn_model.build_model_discrim_sep(FEAT_SHAPE, feat_len, model_type)
        #Load previous model and history
        model.load_weights(PATH_MODEL_CHECKPOINT)
        #----Extract specific weigths----
        weights_list = model.get_weights() 

        #load features (Eval)
        PATH_FEAT_EVAL_I = "./feat/" + input_name + ".feat.eval"
        PATH_FEAT_EVAL_O = "./feat/" + target_name + ".feat.eval"
        f_event_feat_eval_I = open(PATH_FEAT_EVAL_I, 'rb')
        f_event_feat_eval_O = open(PATH_FEAT_EVAL_O, 'rb')
        Y = np.load(f_event_feat_eval_I)
        X = np.load(f_event_feat_eval_O)
        Y = Y[:,1:]
        X = X[:,1:]
        [Y_3D,L1_y] = F.overlap_spec_3D(Y, 64, 128, 0, norm='MAX')
        [X_3D,_] = F.overlap_spec_3D(X, 64, 128, 0, norm='MAX',norm_given=L1_y)
        [i, t, f] = Y_3D.shape
        Y = np.reshape(Y_3D, (i, t, f, 1))
        X = np.reshape(X_3D, (i, t, f, 1))
        print("Finished loading evaluation dataset")

        #Conduct inference
        [b,t,f,c]=Y.shape
        X_hat = np.zeros((b,t,f,c))
        for i in range(0,b):
            [tmp] = model.predict(np.reshape(Y[i,:,:,:], (1,t,f,c)), batch_size=BATCH_SIZE_TEST)
            X_hat[i,:,:,:] = np.reshape(tmp, (1,t,f,c))
        

        #Feat conversion from log-pow to mag
        if feat_type == 'STFT_pow':
            Y = np.sqrt(np.exp(Y)-1)
            X = np.sqrt(np.exp(X)-1)
            X_hat = np.sqrt(np.exp(X_hat)-1)


        #Evaluate results
        [score_x, mae_x] = model.evaluate(X, X_hat, batch_size=BATCH_SIZE_TEST)
        #[score_d, mae_d] = model.evaluate(D, D_hat, batch_size=BATCH_SIZE_TEST)
        rmse_x = np.sqrt(np.mean((np.reshape(X_hat, (1, b*t*f*c)) - np.reshape(X, (1, b*t*f*c))) ** 2))
        
        print('RMSE_x:' + str(rmse_x))   # Printing RMSE
        print('Score_x:' + str(score_x))
        print('MAE_x:' + str(mae_x))
        #print('RMSE_d:' + str(rmse_d))   # Printing RMSE
        #print('Score_d:' + str(score_d))
        #print('MAE_d:' + str(mae_d))
    #
        #Save objective scores to txt
        PATH_RESULT = "./output/" + input_name + "." + model_type + ".result.txt"
    #
        with open(PATH_RESULT, 'w') as fout:
            fout.write('RMSE_x:' + str(rmse_x))
            fout.write('\nScore_x:' + str(score_x))
            fout.write('\nMAE_x:' + str(mae_x))

        #Show Sample Spectrogram as evaluation
        PATH_PIC = "./output/" + input_name + "." + model_type
        Y = np.reshape(Y[1,:,:,:], (t,f))
        wavPlot.spectrogram(Y.T, PATH_PIC + ".Y.sample")
        X = np.reshape(X[1,:,:,:], (t,f))
        wavPlot.spectrogram(X.T, PATH_PIC + ".X.sample")
        X_hat = np.reshape(X_hat[1,:,:,:], (t,f))
        wavPlot.spectrogram(X_hat.T, PATH_PIC + ".X_hat.sample")

    if model_type == 'TAU-Net' or model_type == 'TAU-Net2':

        timestep = 32
        FEAT_SHAPE = (timestep, feat_len-1, 1)

        #----Define model structure then compile-----
        #----Load NMF Dictionary----
        PATH_AUXMODEL_X = 'model_NMF/GC_GIST_rec2_speech.SNMF_FULLBATCH.basis'
        PATH_AUXMODEL_D = 'model_NMF/GC_GIST_rec2_DI.SNMF_FULLBATCH.basis'
        B_x = np.loadtxt(PATH_AUXMODEL_X)
        B_d = np.loadtxt(PATH_AUXMODEL_D)
        B_x = np.reshape(B_x[1:,:], (1,512,64))
        B_d = np.reshape(B_d[1:,:], (1,512,64))
        B_x = np.tile(B_x, (BATCH_SIZE_TEST, 1, 1))
        B_d = np.tile(B_d, (BATCH_SIZE_TEST, 1, 1))

        [model, encoder, decoder] = nn_model.build_model_dae(FEAT_SHAPE, model_type, B_x, B_d, BATCH_SIZE_TEST, timestep)
        
        #Load previous model and history
        model.load_weights(PATH_MODEL_CHECKPOINT)
        ##encoder.load_weights(PATH_MODEL_CHECKPOINT_ENC)
        ##decoder.load_weights(PATH_MODEL_CHECKPOINT_DEC)        
        #----Extract specific weigths----
        weights_list = model.get_weights() 

        #load features (Eval)
        PATH_FEAT_EVAL_I = "./feat/" + input_name + ".feat.eval"
        PATH_FEAT_EVAL_O = "./feat/" + target_name + ".feat.eval"
        f_event_feat_eval_I = open(PATH_FEAT_EVAL_I, 'rb')
        f_event_feat_eval_O = open(PATH_FEAT_EVAL_O, 'rb')
        Y = np.load(f_event_feat_eval_I)
        X = np.load(f_event_feat_eval_O)
        Y = Y[:,1:]
        X = X[:,1:]
        [Y_3D,L1_y] = F.overlap_spec_3D(Y, int(timestep/2), int(1), 0, norm='MAX')
        [X_3D,_] = F.overlap_spec_3D(X, int(timestep/2), int(1), 0, norm='MAX', norm_given=L1_y)
        [i, t, f] = Y_3D.shape
        Y = np.reshape(Y_3D, (i, t, f, 1))
        X = np.reshape(X_3D, (i, t, f, 1))

        print("Finished loading evaluation dataset")

        #Conduct inference
        [b,t,f,c]=Y.shape
        b = 10
        IMG_IDX = 6
        X_hat = np.zeros((b,t,f,c))
        for i in range(0,b):
            y_in = np.reshape(Y[i,:,:,:], (1,t,f,c))
            [tmp, _,_, _]=model.predict(y_in, batch_size=BATCH_SIZE_TEST)
            ##enc_out = encoder.predict(y_in, batch_size=BATCH_SIZE_TEST)
            ##if model_type == 'TAU-Net2':
            ##    [tmp, tmp_d] = decoder.predict([y_in, *enc_out], batch_size=BATCH_SIZE_TEST)
            ##else:
            ##    tmp = decoder.predict([y_in, *enc_out], batch_size=BATCH_SIZE_TEST)
            
            #X_hat[i,:] = tmp[8,:]
            #X_hat[i,:] = np.reshape(tmp_NMF, (1,f))
            #tmp = model.predict(y_in, batch_size=BATCH_SIZE_TEST)
            X_hat[i,:,:,:] = np.reshape(tmp, (1,t,f,c))
        ##wavPlot.spectrogram(X_hat.T, None)
        #Evaluate results
        #[score_x, mae_x] = decoder.evaluate([X], X_hat, batch_size=BATCH_SIZE_TEST)
        #[score_d, mae_d] = model.evaluate(D, D_hat, batch_size=BATCH_SIZE_TEST)
        ##mae_x = np.mean(np.abs((np.reshape(X_hat[:b,:,:,:], (1, b*f*t)) - np.reshape(X[:b,:,:,:], (1, b*f*t)))))
        ##rmse_x = np.sqrt(np.mean((np.reshape(X_hat[:b,:,:,:], (1, b*f*t)) - np.reshape(X[:b,:,:,:], (1, b*f*t))) ** 2))
        
        ##print('RMSE_x:' + str(rmse_x))   # Printing RMSE
        ##print('Score_x:' + str(score_x))
        ##print('MAE_x:' + str(mae_x))
        #print('RMSE_d:' + str(rmse_d))   # Printing RMSE
        #print('Score_d:' + str(score_d))
        #print('MAE_d:' + str(mae_d))
    #
        #Save objective scores to txt
        PATH_RESULT = "./output/" + input_name + "." + model_type + ".result.txt"
    #
        ##with open(PATH_RESULT, 'w') as fout:
        ##    fout.write('RMSE_x:' + str(rmse_x))
        ##    ##fout.write('\nScore_x:' + str(score_x))
        ##    fout.write('\nMAE_x:' + str(mae_x))

        #Show Sample Spectrogram as evaluation
        PATH_PIC = "./output/" + input_name + "." + model_type
        Y = np.reshape(Y[IMG_IDX*int(128/32):(IMG_IDX+1)*int(128/32),:,:,:], (128,f))
        wavPlot.spectrogram(Y.T, PATH_PIC + ".Y.sample")
        X = np.reshape(X[IMG_IDX*int(128/32):(IMG_IDX+1)*int(128/32),:,:,:], (128,f))
        wavPlot.spectrogram(X.T, PATH_PIC + ".X.sample")
        X_hat = np.reshape(X_hat[IMG_IDX*int(128/32):(IMG_IDX+1)*int(128/32),:,:,:], (128,f))
        wavPlot.spectrogram(X_hat.T, PATH_PIC + ".X_hat.sample")       


def eval_model_embedding(input_name, source_names, num_embed, num_source, model_type):
    if CPU_MODE == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

    #Load feature info
    PATH_PICKLE = "./feat/" + input_name + ".feat.pickle"
    with open(PATH_PICKLE, 'rb') as f_event_info:
        [input_name, DB_path, DB_read_seed, num_file_DB, feat_type, \
        fs, flen, foverlap, nFFT, DC_Bin, feat_len] = pickle.load(f_event_info)

    PATH_MODEL_CHECKPOINT = "./model/" + input_name + "." + model_type + ".best.model"
    

    #----Reconstruct Data from Model (Evaluation)----
    if model_type == 'CDE-Net' or model_type == 'DeepCluster':

        timestep = 128
        FEAT_SHAPE = (timestep, feat_len-1, 1)

        #----Define model structure then compile-----
        #----Load NMF Dictionary----
        ##PATH_AUXMODEL_X = 'model_nmf/DS_10283_noisy_clean.SNMF_FULLBATCH.basis'
        ##PATH_AUXMODEL_D = 'model_nmf/DS_10283_noisy_noise.SNMF_FULLBATCH.basis'
        ##B_x = np.loadtxt(PATH_AUXMODEL_X)
        ##B_d = np.loadtxt(PATH_AUXMODEL_D)
        ##B_x = B_x[1:,:]
        ##B_d = B_d[1:,:]
        
        model = nn_model.build_model_embedding(FEAT_SHAPE, model_type, num_embed, num_source, BATCH_SIZE_TEST, timestep)

        #Load previous model and history
        model.load_weights(PATH_MODEL_CHECKPOINT)
        #----Extract specific weigths----
        weights_list = model.get_weights() 

        #----Load Features-----
        #load features (Eval)
        PATH_FEAT_TRAIN_I = "./feat/" + input_name + ".feat.eval"
        f_event_feat_train_I = open(PATH_FEAT_TRAIN_I, 'rb')
        Y = np.load(f_event_feat_train_I)
        Y = Y[:,1:]
        [Y_3D,_] = F.overlap_spec_3D(Y, int(timestep/2), int(timestep/4), 0, norm='L1')
        [i, t, f] = Y_3D.shape
        Y = np.reshape(Y_3D, (i, t, f, 1))
        X = np.zeros((i, t, f, num_source))
        #M = np.zeros((i, t, f, num_source))
        for c in range(0,num_source):
            PATH_FEAT_TRAIN_X = "./feat/" + source_names[c] + ".feat.eval"
            f_event_feat_train_X = open(PATH_FEAT_TRAIN_X, 'rb')
            X_c = np.load(f_event_feat_train_X)
            X_c = X_c[:,1:]
            [X_3D,_] = F.overlap_spec_3D(X_c, int(timestep/2), int(timestep/4), 0, norm=None)
            [i, t, f] = X_3D.shape
            X_c = np.reshape(X_3D, (i, t, f, 1))
            X[:,:,:,c] = X_c[:,:,:,0]
            #N = np.abs(Y - X_c)
            #M_tmp = get_IBM(X_c, N, LC=0)
            #M[:,:,:,c] = M_tmp
        
        X = X[:,:,:,0] #First source: Speech
        print("Finished loading evaluation dataset")

        #Conduct inference
        [b,t,f,c]=Y.shape
        I = num_embed
        C = num_source
        B = int(128 / timestep * 3)
        M_hat = np.zeros((B,t,f,C))
        X_hat = np.zeros((B,t,f,C))
        X_irm = np.zeros((B,t,f,1))
        for i in range(0,B):
            y_in = np.reshape(Y[i,:,:,:], (1,t,f,1))
            ####y_in = y_in * y_in #pow feature
            V = model.predict(y_in, batch_size=BATCH_SIZE_TEST)
            V = np.resize(np.squeeze(V), (t*f, I))
            wavPlot.spectrogram(np.matmul(V.T, V), None)
            #----K-means clustering----
            kmeans = KMeans(C)
            M_kmeans = kmeans.fit(V.T)
            M_labels = kmeans.labels_
            M_labels = np.reshape(M_labels, (I,1))
            Attention = np.concatenate([(1-M_labels), M_labels], axis=1)
            M_kmeans = np.matmul(V, Attention)
            M_kmeans /= np.max(np.max(M_kmeans))
            #M_kmeans = M_kmeans.cluster_centers_.T
            M_hat[i,:,:,:] = np.reshape(M_kmeans, (1,t,f,C))
            X_hat[i,:,:,0] = np.squeeze(y_in) * (M_hat[i,:,:,0])
            ##X_irm[i,:,:,0] = np.squeeze(X_tmp)
            print(str(i) + 'th block clustered')

        X = X[0:B,:,:]
        X_hat = X_hat[:,:,:,0] #First source: Speech
        #Evaluate results
        #[score_x, mae_x] = decoder.evaluate([X], X_hat, batch_size=BATCH_SIZE_TEST)
        #[score_d, mae_d] = model.evaluate(D, D_hat, batch_size=BATCH_SIZE_TEST)
        mae_x = np.mean(np.abs((np.reshape(X_hat, (1, B*t*f)) - np.reshape(X, (1, B*t*f)))))
        rmse_x = np.sqrt(np.mean((np.reshape(X_hat, (1, B*t*f)) - np.reshape(X, (1, B*t*f))) ** 2))
        
        print('RMSE_x:' + str(rmse_x))   # Printing RMSE
        ##print('Score_x:' + str(score_x))
        print('MAE_x:' + str(mae_x))
        #print('RMSE_d:' + str(rmse_d))   # Printing RMSE
        #print('Score_d:' + str(score_d))
        #print('MAE_d:' + str(mae_d))
    #
        #Save objective scores to txt
        PATH_RESULT = "./output/" + input_name + "." + model_type + ".result.txt"
    #
        with open(PATH_RESULT, 'w') as fout:
            fout.write('RMSE_x:' + str(rmse_x))
            ##fout.write('\nScore_x:' + str(score_x))
            fout.write('\nMAE_x:' + str(mae_x))

        #Show Sample Spectrogram as evaluation
        PATH_PIC = "./output/" + input_name + "." + model_type
        Y = np.reshape(Y[1:int(128/timestep)+1,:,:,:], (128,f))
        wavPlot.spectrogram(Y.T, PATH_PIC + ".Y.sample")
        X = np.reshape(X[1:int(128/timestep)+1,:,:], (128,f))
        wavPlot.spectrogram(X.T, PATH_PIC + ".X.sample")
        X_hat = np.reshape(X_hat[1:int(128/timestep)+1,:,:], (128,f))
        wavPlot.spectrogram(X_hat.T, PATH_PIC + ".X_hat.sample")        
        ##X_irm = np.reshape(X_irm[0,:,:], (t,f))
        ##wavPlot.spectrogram(X_irm.T, PATH_PIC + ".X_irm.sample")     
