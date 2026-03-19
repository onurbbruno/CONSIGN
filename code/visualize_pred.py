import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from helpers_cp import *
from config_cp import Config
from PW_conformal_prediction import PW_calibration,PW_metrics, PW_plot
from SVD_conformal_prediction_UQ import SVD_calibration,SVD_preprocess,SVD_metrics, SVD_plot

lambdas = extract_lambdas(f"metrics/{Config.challenge}_{Config.alpha}_{Config.beta}_{Config.K_max}.txt")
# exctract softmax
smx,labels,imgs, _, v_splits = extract_softmax()

lambdas_PW = lambdas[:,1]
lambdas_SVD = lambdas[:,0]
lambdas_SACP = lambdas[:,2]

# loop for different validation splits
for cal,v_split in enumerate(v_splits):

    val_smx = smx[v_split,:]
    val_labels = labels[v_split,:]
    val_imgs = imgs[v_split,:]

    # extract calibration lambda
    lam_PW = lambdas_PW[cal]
    lam_SVD = lambdas_SVD[cal]
    lam_SACP = lambdas_SACP[cal]
    # test hyeperparameters
    n_test = val_smx.shape[0]

    # pre stuff for SVD
    softmax_mean = np.mean(val_smx,axis=1)
    a_bound,b_bound,U,Sig,mean_sample = SVD_preprocess(val_smx)

    for i in range(n_test):
        X = softmax_mean[i,:]
        Y = val_labels[i,:]
        U_test = U[i,:]
        Sig_test = Sig[i,:]

        samples_PW = PW_plot(lam_PW,X,Config.pw_strategy)

        samples_SACP = PW_plot(lam_PW,X,Config.pw_strategy,'SACP')

        samples_SVD = SVD_plot(lam_SVD,a_bound[i,:],b_bound[i,:],U_test,mean_sample[i,:],Sig_test)

        if Config.challenge in ['MnM2','mscmr19']:
            heart_visual(val_imgs[i,0,0,:],samples_PW,samples_SVD,samples_SACP,Y)
        elif Config.challenge == 'LIDC':
            LIDC_visual(val_imgs[i,:],samples_PW,samples_SVD,samples_SACP,Y)
        else:
            COCO_visual(val_imgs[i,:],samples_PW,samples_SVD,samples_SACP,Y)
