import wavLearn
import wavFeat
#import wavEval
import argparse
import json
import numpy as np

def main(param):

    INPUT_NAME = param['path']['INPUT_NAME']
    TARGET_NAME = param['path']['TARGET_NAME']
    NOISE_NAME = param['path']['NOISE_NAME']
    INTERF_NAME = param['path']['INTERF_NAME']
    INPUT_PATH = param['path']['INPUT_PATH']
    TARGET_PATH = param['path']['TARGET_PATH']
    NOISE_PATH = param['path']['NOISE_PATH']
    INTERF_PATH = param['path']['INTERF_PATH']
    PATH_AUXMODEL_X = param['path']['PATH_AUXMODEL_X']
    PATH_AUXMODEL_D = param['path']['PATH_AUXMODEL_D']
    MODEL_TYPE = param['path']['MODEL_TYPE']
    TIME_STEP = param['nn']['timestep']
    FEAT_EXT = param['feat']['feat_ext']
    if MODEL_TYPE == 'TAU-Net':
        #Extract INPUT_PATH features
        if FEAT_EXT == 1:
                wavFeat.feat_ext(INPUT_NAME, INPUT_PATH, param)
                wavFeat.feat_ext(TARGET_NAME, TARGET_PATH, param)
                wavFeat.feat_ext(NOISE_NAME, NOISE_PATH, param)
                wavFeat.feat_ext(INTERF_NAME, INTERF_PATH, param)

        #Learn Speech basis
        #wavLearn.train_model_NMF(TARGET_NAME, 'SNMF_FULLBATCH', 64, param)
        #wavEval.eval_model_NMF(TARGET_NAME, 'SNMF_FULLBATCH', param)

        #Learn Noise basis
        #wavLearn.train_model_NMF(NOISE_NAME, 'SNMF_FULLBATCH', 64, param)
        #wavEval.eval_model_NMF(NOISE_NAME, 'SNMF_FULLBATCH', param)

        #Learn Interferer basis
        #wavLearn.train_model_NMF(INTERF_NAME, 'SNMF_FULLBATCH', 32, param)
        #wavEval.eval_model_NMF(NOISE_NAME, 'SNMF_FULLBATCH', param)
        
        #Merge Noise and Interfer bases as one
        ##PATH_MODEL_D = "./model_NMF/" + NOISE_NAME + "." + 'SNMF_FULLBATCH.basis'
        ##PATH_MODEL_I = "./model_NMF/" + INTERF_NAME + "." + 'SNMF_FULLBATCH.basis'
        ##PATH_MODEL_DI = "./model_NMF/" + INPUT_NAME + "_DI" + "." + 'SNMF_FULLBATCH.basis'
        ##B_d = np.loadtxt(PATH_MODEL_D)
        ##B_i = np.loadtxt(PATH_MODEL_I)
        ##B_d = np.concatenate([B_d,B_i],axis=-1)
        ##np.savetxt(PATH_MODEL_DI, B_d, delimiter='   ', fmt='%.7e')

        #Learn TAU Net
        wavLearn.train_model_GC(INPUT_NAME, TARGET_NAME, NOISE_NAME, INTERF_NAME, MODEL_TYPE, PATH_AUXMODEL_X, PATH_AUXMODEL_D, param)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--configs", type=str, default="configs/TAUnet_GC_sim_data5.json", help="load json config file")
    args = vars(ap.parse_args())

    json_param = open(args["configs"]).read()
    param = json.loads(json_param)
    main(param)
